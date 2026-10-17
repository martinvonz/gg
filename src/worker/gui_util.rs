//! Analogous to cli_util from jj-cli
//! We reuse a bit of jj-cli code, but many of its modules include TUI concerns or are not suitable for a long-running server

use std::{
    cell::OnceCell,
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
    slice,
    sync::Arc,
};

use anyhow::{Context, Result, anyhow};
use futures_util::TryStreamExt;
use itertools::Itertools;
use jj_frontend::{
    revset_util::UserRevsetEvaluationError,
    runner::{
        RunnerError, RunnerEvent, RunnerEvents, RunnerOptions, SnapshotConfig,
        SnapshotConfigDiagnostics, WorkspaceOperationRunner,
    },
    workspace_env::{WorkspaceEnvironment, WorkspaceEnvironmentDiagnostics, find_workspace_dir},
};
use jj_lib::{
    backend::{BackendError, ChangeId, CommitId},
    commit::Commit,
    default_backend_factories,
    default_index::DefaultReadonlyIndex,
    file_util,
    fileset::FilesetAliasesMap,
    git::{self, REMOTE_NAME_FOR_LOCAL_GIT_REPO},
    git_backend::GitBackend,
    id_prefix::{IdPrefixContext, IdPrefixIndex},
    object_id::ObjectId,
    ref_name::{WorkspaceName, WorkspaceNameBuf},
    repo::{ReadonlyRepo, Repo},
    repo_path::RepoPath,
    revset::{
        Revset, RevsetAliasesMap, RevsetDiagnostics, RevsetEvaluationError, RevsetExpression,
        RevsetExtensions, RevsetResolutionError, RevsetStreamExt, UserRevsetExpression,
    },
    rewrite,
    settings::UserSettings,
    transaction::Transaction,
    view::View,
    workspace::{DefaultWorkspaceLoaderFactory, Workspace, WorkspaceLoaderFactory},
};
use pollster::FutureExt as _;
use thiserror::Error;

use super::{WorkerSession, git_util::get_git_remote_names};

use crate::{
    config::{self, GGSettings, WorkspaceLocation, read_config},
    messages::{self, *},
};

/// Loaded state of the worker thread, with borrowed JJ data.
///
/// Create one with [`WorkerSession::load_workspace()`], or use
/// [`super::session::SessionEvent`]s in a dedicated thread to manage
/// the state machine.
///
/// See the [crate-level docs](crate) for a usage overview.
pub struct WorkspaceSession<'a> {
    pub session: &'a mut WorkerSession,

    // workspace-level data, initialised once (and again when the config changes)
    runner: WorkspaceOperationRunner,
    pub(crate) data: WorkspaceData,
    snapshot_config: SnapshotConfig,
    is_large: bool, // this is based on the head operation and thus derived from the rest of the data
    is_colocated: bool, // theoretically operation-specific but we don't support switching horses

    // operation-specific data, containing a repo view and derived extras
    operation: OperationData,
}

pub(crate) struct WorkspaceData {
    pub workspace_settings: UserSettings,
    pub query_choices: HashMap<String, String>,
}

/// state derived from a specific operation
pub(crate) struct OperationData {
    pub repo: Arc<ReadonlyRepo>,
    pub wc_id: CommitId,
    ref_index: OnceCell<Rc<RefIndex>>,
    workspace_index: OnceCell<HashMap<CommitId, String>>,
    prefix_context: IdPrefixContext,
}

#[derive(Debug, Error)]
pub(crate) enum RevsetError {
    #[error(transparent)]
    Resolution(#[from] RevsetResolutionError),
    #[error(transparent)]
    Evaluation(#[from] RevsetEvaluationError),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<UserRevsetEvaluationError> for RevsetError {
    fn from(value: UserRevsetEvaluationError) -> Self {
        match value {
            UserRevsetEvaluationError::Resolution(err) => RevsetError::Resolution(err),
            UserRevsetEvaluationError::Evaluation(err) => RevsetError::Evaluation(err),
        }
    }
}

impl From<BackendError> for RevsetError {
    fn from(value: BackendError) -> Self {
        RevsetError::Other(anyhow!(value))
    }
}

impl WorkerSession {
    pub async fn load_workspace(&mut self, cwd: &Path) -> Result<WorkspaceSession<'_>> {
        let factory = DefaultWorkspaceLoaderFactory;
        let loader = factory.create(find_workspace_dir(cwd))?;

        let (settings, aliases_map, fileset_aliases_map, preset_choices) =
            read_config(Some(&WorkspaceLocation {
                workspace_root: loader.workspace_root().to_owned(),
                repo_path: loader.repo_path().to_owned(),
            }))?;

        let workspace = loader.load(
            &settings,
            &default_backend_factories::default_backend_factories(),
            &default_backend_factories::default_working_copy_factories(),
        )?;

        let data: WorkspaceData = WorkspaceData {
            workspace_settings: settings,
            query_choices: preset_choices,
        };

        let env = new_workspace_environment(&workspace, aliases_map, fileset_aliases_map)?;
        let snapshot_config = load_snapshot_config(&workspace, &env)?;
        let runner = WorkspaceOperationRunner::load_at_head(
            &LogRunnerEvents,
            workspace,
            env,
            runner_options(),
        )
        .await?;
        let operation =
            OperationData::new(runner.workspace_name(), runner.env(), runner.repo().clone());
        let workspace = runner.workspace();

        let index_store = workspace.repo_loader().index_store();
        let index = index_store
            .get_index_at_op(operation.repo.operation(), workspace.repo_loader().store())
            .await?;
        let is_large = if let Some(default_index) = index.downcast_ref::<DefaultReadonlyIndex>() {
            let stats = default_index.stats();
            stats.num_commits as i64 >= data.workspace_settings.query_large_repo_heuristic()
        } else {
            true
        };

        let is_colocated = runner.env().working_copy_shared_with_git();

        Ok(WorkspaceSession {
            session: self,
            runner,
            data,
            snapshot_config,
            is_large,
            is_colocated,
            operation,
        })
    }
}

impl WorkspaceSession<'_> {
    pub fn name(&self) -> &WorkspaceName {
        self.runner.workspace_name()
    }

    pub(crate) fn workspace(&self) -> &Workspace {
        self.runner.workspace()
    }

    pub(crate) fn env(&self) -> &WorkspaceEnvironment {
        self.runner.env()
    }

    /// Re-reads the configuration after it has been changed. The workspace is reloaded so that it
    /// uses the new settings.
    pub(crate) fn reload_config(&mut self) -> Result<()> {
        let workspace_root = self.runner.workspace_root().to_owned();
        let loader = DefaultWorkspaceLoaderFactory.create(&workspace_root)?;
        let aliases_map;
        let fileset_aliases_map;
        (
            self.data.workspace_settings,
            aliases_map,
            fileset_aliases_map,
            self.data.query_choices,
        ) = read_config(Some(&WorkspaceLocation {
            workspace_root,
            repo_path: loader.repo_path().to_owned(),
        }))?;
        let workspace = loader.load(
            &self.data.workspace_settings,
            &default_backend_factories::default_backend_factories(),
            &default_backend_factories::default_working_copy_factories(),
        )?;
        let env = new_workspace_environment(&workspace, aliases_map, fileset_aliases_map)?;
        self.snapshot_config = load_snapshot_config(&workspace, &env)?;
        let repo = self.runner.repo().clone();
        self.runner = WorkspaceOperationRunner::load_at(
            &LogRunnerEvents,
            workspace,
            repo,
            env,
            runner_options(),
            true,
        )?;
        self.operation =
            OperationData::new(self.name(), self.runner.env(), self.runner.repo().clone());
        Ok(())
    }

    /// Updates the data derived from the current operation if the runner has moved to a new
    /// operation. Returns whether it had.
    fn sync_operation(&mut self) -> bool {
        if self.runner.repo().op_id() == self.operation.repo.op_id() {
            return false;
        }
        self.operation =
            OperationData::new(self.name(), self.runner.env(), self.runner.repo().clone());
        true
    }

    pub fn wc_id(&self) -> &CommitId {
        &self.operation.wc_id
    }

    pub async fn add_workspace(&mut self, name: String, path: PathBuf) -> Result<()> {
        let workspace_name: WorkspaceNameBuf = name.into();

        anyhow::ensure!(
            !workspace_name.as_str().is_empty(),
            "workspace name cannot be empty"
        );
        anyhow::ensure!(
            self.view().get_wc_commit_id(&workspace_name).is_none(),
            "workspace '{}' already exists",
            workspace_name.as_symbol()
        );
        if path.exists() {
            anyhow::ensure!(
                file_util::is_empty_dir(&path)?,
                "destination path exists and is not an empty directory"
            );
        } else {
            std::fs::create_dir(&path)
                .with_context(|| format!("failed to create directory {}", path.display()))?;
        }

        // snapshot current workspace before creating the new one
        self.import_and_snapshot(true, false).await?;

        let (mut new_workspace, repo) = Workspace::init_workspace_with_existing_repo(
            &path,
            self.workspace().repo_path(),
            &self.operation.repo,
            &*default_backend_factories::default_working_copy_factory(),
            workspace_name.clone(),
        )
        .await?;

        // set up the real WC commit for the new workspace
        let mut tx = repo.start_transaction();

        let wc_commit = tx.repo().store().get_commit(self.wc_id())?;
        let parents = wc_commit.parents().await?;
        let parent_ids: Vec<_> = parents.iter().map(|c| c.id().clone()).collect();
        let tree = rewrite::merge_commit_trees(tx.repo(), &parents).await?;
        let new_wc_commit = tx.repo_mut().new_commit(parent_ids, tree).write().await?;
        tx.repo_mut()
            .edit(workspace_name.clone(), &new_wc_commit)
            .await?;

        self.finish_transaction(
            tx,
            format!(
                "create initial working-copy commit in workspace '{}'",
                workspace_name.as_symbol()
            ),
        )
        .await?;

        new_workspace
            .check_out(self.operation.repo.op_id().clone(), None, &new_wc_commit)
            .await?;

        Ok(())
    }

    pub async fn forget_workspace(&mut self, name: String) -> Result<()> {
        let workspace_name: WorkspaceNameBuf = name.into();

        anyhow::ensure!(
            self.view().get_wc_commit_id(&workspace_name).is_some(),
            "workspace '{}' not found",
            workspace_name.as_symbol()
        );
        anyhow::ensure!(
            *workspace_name != *self.name(),
            "cannot forget the current workspace"
        );

        let mut tx = self.start_transaction().await?;
        tx.repo_mut().remove_workspace(&workspace_name).await?;

        let workspace_store = tx.base_repo().loader().workspace_store();
        workspace_store.forget(&[&*workspace_name])?;

        self.finish_transaction(
            tx,
            format!("forget workspace '{}'", workspace_name.as_symbol()),
        )
        .await?;

        Ok(())
    }

    pub async fn rename_workspace(
        &mut self,
        old_name: WorkspaceNameBuf,
        new_name: WorkspaceNameBuf,
    ) -> Result<Option<messages::RepoStatus>> {
        let mut tx = self.start_transaction().await?;

        // capture before locking the working copy
        let workspace_store = tx.base_repo().loader().workspace_store().clone();
        let is_colocated = self.is_colocated;

        let (mut locked_ws, _) = self.runner.start_working_copy_mutation().await?;

        locked_ws.locked_wc().rename_workspace(new_name.clone());
        tx.repo_mut()
            .rename_workspace(&old_name, new_name.clone())?;

        workspace_store.rename(&old_name, &new_name)?;

        if is_colocated {
            git::export_refs(tx.repo_mut())?;
        }

        let repo = tx
            .commit(format!(
                "rename workspace '{}' to '{}'",
                old_name.as_symbol(),
                new_name.as_symbol()
            ))
            .await?;
        locked_ws.finish(repo.op_id().clone()).await?;

        self.runner.reload_at_head(&LogRunnerEvents).await?;
        self.sync_operation();

        Ok(Some(self.format_status()))
    }

    pub fn workspace_root(&self, name: String) -> Result<PathBuf> {
        let workspace_name: WorkspaceNameBuf = name.into();

        anyhow::ensure!(
            self.view().get_wc_commit_id(&workspace_name).is_some(),
            "Workspace '{}' not found",
            workspace_name.as_symbol()
        );

        // the loaded workspace knows where it is without consulting the store
        if *workspace_name == *self.name() {
            return absolute_workspace_path(self.workspace().workspace_root());
        }

        let repo_path = self.workspace().repo_path();
        let workspace_store = self.operation.repo.loader().workspace_store();

        match workspace_store.get_workspace_path(&workspace_name)? {
            // stored paths are relative to the repo directory
            Some(stored_path) => absolute_workspace_path(&repo_path.join(stored_path)),
            // repos created before jj tracked workspace paths have no entries, but
            // the default workspace is the one hosting the repo, so we can find it
            None if *workspace_name == *WorkspaceName::DEFAULT => repo_path
                .parent()
                .and_then(Path::parent)
                .filter(|root| hosts_repo(root, repo_path))
                .map(absolute_workspace_path)
                .unwrap_or_else(|| Err(anyhow!("Workspace 'default' has no recorded path"))),
            None => Err(anyhow!(
                "Workspace '{}' has no recorded path",
                workspace_name.as_symbol()
            )),
        }
    }

    pub fn list_workspaces(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .view()
            .wc_commit_ids()
            .keys()
            .map(|name| name.as_symbol().to_string())
            .collect();
        names.sort();
        names
    }

    pub(crate) fn sink(&self) -> Arc<dyn super::EventSink> {
        self.session.sink.clone()
    }

    // XXX maybe: hunt down uses and make nonpub
    pub(crate) fn repo(&self) -> &ReadonlyRepo {
        self.operation.repo.as_ref()
    }

    pub(crate) fn view(&self) -> &View {
        self.operation.repo.view()
    }

    pub(crate) fn get_commit(&self, id: &CommitId) -> Result<Commit> {
        Ok(self.operation.repo.store().get_commit(id)?)
    }

    pub(crate) fn git_repo(&self) -> Option<gix::Repository> {
        self.operation
            .git_backend()
            .map(|backend| backend.git_repo().to_owned())
    }

    pub(crate) async fn load_at_head(&mut self) -> Result<bool> {
        self.runner.reload_at_head(&LogRunnerEvents).await?;
        Ok(self.sync_operation())
    }

    /***********************************************************/
    /* Functions for evaluating revset expressions             */
    /* unfortunately parse_context and resolver are not cached */
    /***********************************************************/

    pub(crate) fn evaluate_revset_expr<'op>(
        &'op self,
        repo: &'op dyn Repo,
        revset_expr: Arc<UserRevsetExpression>,
    ) -> Result<Box<dyn Revset + 'op>, RevsetError> {
        let revset = self
            .env()
            .revset_evaluator(repo, &self.operation.prefix_context, revset_expr)
            .evaluate()?;
        Ok(revset)
    }

    pub(crate) fn evaluate_revset_str<'op>(
        &'op self,
        revset_str: &str,
    ) -> Result<Box<dyn Revset + 'op>, RevsetError> {
        let revset_expr = self
            .env()
            .parse_revset(&mut RevsetDiagnostics::new(), revset_str)
            .context("parse revset")?;
        self.evaluate_revset_expr(self.operation.repo.as_ref(), revset_expr)
    }

    pub(crate) fn evaluate_revset_commits<'op>(
        &'op self,
        ids: &[messages::CommitId],
    ) -> Result<Box<dyn Revset + 'op>, RevsetError> {
        let expr = RevsetExpression::commits(
            ids.iter()
                .map(|id| CommitId::try_from_hex(id.hex.as_str()).expect("frontend-validated id"))
                .collect(),
        );
        self.evaluate_revset_expr(self.operation.repo.as_ref(), expr)
    }

    pub(crate) fn evaluate_revset_changes<'op>(
        &'op self,
        ids: &[messages::ChangeId],
    ) -> Result<Box<dyn Revset + 'op>, RevsetError> {
        let mut expr = RevsetExpression::none();
        for id in ids.iter() {
            expr = expr.union(&RevsetExpression::symbol(id.hex.clone()))
        }
        self.evaluate_revset_expr(self.operation.repo.as_ref(), expr)
    }

    pub(crate) fn evaluate_immutable(&self) -> Result<Box<dyn Revset + '_>> {
        let repo = self.operation.repo.as_ref();
        let expr = self.env().resolve_immutable_expression(repo, false)?;
        Ok(expr.evaluate(repo)?)
    }

    fn resolve_optional<'op, 'set: 'op, T: AsRef<dyn Revset + 'set>>(
        &'op self,
        revset: T,
    ) -> Result<Option<Commit>, RevsetError> {
        let store = self.operation.repo.store();
        let mut stream = revset.as_ref().stream().commits(store);
        let first = stream.try_next().block_on()?;
        let second = stream.try_next().block_on()?;
        match (first, second) {
            (Some(commit), None) => Ok(Some(commit)),
            (None, _) => Ok(None),
            (Some(_), Some(_)) => Err(RevsetError::Other(anyhow!(
                r#"Revset "{:?}" resolved to more than one revision"#,
                revset.as_ref()
            ))),
        }
    }

    fn resolve_single<'op, 'set: 'op, T: AsRef<dyn Revset + 'set>>(
        &'op self,
        revset: T,
    ) -> Result<Commit, RevsetError> {
        match self.resolve_optional(revset)? {
            Some(commit) => Ok(commit),
            None => Err(RevsetError::Other(anyhow!(
                "Revset didn't resolve to any revisions"
            ))),
        }
    }

    fn format_id_str(id: &RevId) -> String {
        match id.change.offset {
            Some(offset) => format!("{}/{}", id.change.hex, offset),
            _ => id.change.hex.clone(),
        }
    }

    // policy: queries operate on a precise change using offsets, with explicit not-found results
    pub(crate) fn resolve_optional_id(&self, id: &RevId) -> Result<Option<Commit>, RevsetError> {
        let id_str = Self::format_id_str(id);
        let change_revset = match self.evaluate_revset_str(&id_str) {
            Ok(revset) => revset,
            Err(RevsetError::Resolution(RevsetResolutionError::NoSuchRevision { .. })) => {
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        let store = self.operation.repo.store();
        let mut change_stream = change_revset.as_ref().stream().commits(store);
        let first = change_stream.try_next().block_on()?;
        let second = change_stream.try_next().block_on()?;
        match (first, second) {
            (Some(commit), None) => Ok(Some(commit)),
            (None, _) => Ok(None),
            (Some(_), Some(_)) => {
                let commit_revset = self.evaluate_revset_commits(slice::from_ref(&id.commit))?;
                let mut commit_stream = commit_revset.as_ref().stream().commits(store);
                match commit_stream.try_next().block_on()? {
                    Some(commit) => Ok(Some(commit)),
                    None => Ok(None),
                }
            }
        }
    }

    // policy: queries operate on a precise change using offsets, with explicit not-found results
    pub(crate) fn resolve_optional_set(
        &self,
        set: &RevSet,
    ) -> Result<Option<Vec<Commit>>, RevsetError> {
        let revset_str = format!(
            "{}::{}",
            Self::format_id_str(&set.from),
            Self::format_id_str(&set.to)
        );
        let revset = match self.evaluate_revset_str(&revset_str) {
            Ok(revset) => revset,
            Err(RevsetError::Resolution(RevsetResolutionError::NoSuchRevision { .. })) => {
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        let commits = self.resolve_multiple(revset)?;
        if commits.is_empty() {
            Ok(None)
        } else {
            Ok(Some(commits))
        }
    }

    // policy: most commands prefer to operate on a change and will fail if the change has been evolved
    // however, if it had a known offset they'll use that, and if it's *become* divergent,
    // they will fall back to the known commit so that divergences can be resolved
    pub(crate) fn resolve_change_id(&self, id: &RevId) -> Result<Commit, RevsetError> {
        let id_str = Self::format_id_str(id);
        let revset = self.evaluate_revset_str(&id_str)?;
        let store = self.operation.repo.store();
        let mut stream = revset.as_ref().stream().commits(store);
        let first = stream.try_next().block_on()?;
        let second = stream.try_next().block_on()?;
        let optional_change = match (first, second) {
            (Some(commit), None) => Some(commit),
            (None, _) => None,
            (Some(_), Some(_)) => Some(self.resolve_commit_id(&id.commit)?),
        };

        match optional_change {
            Some(commit) => {
                let resolved_id = commit.id();
                if resolved_id == self.wc_id() || resolved_id.hex().starts_with(&id.commit.prefix) {
                    Ok(commit)
                } else {
                    Err(RevsetError::Other(anyhow!(
                        r#""{}" didn't resolve to the expected commit {}"#,
                        id.change.prefix,
                        id.commit.prefix
                    )))
                }
            }
            None => Err(RevsetError::Other(anyhow!(
                r#""{}" didn't resolve to any revisions"#,
                id.change.prefix
            ))),
        }
    }

    // policy: commands operating on a revset assume it uses precise targeting, including offsets if divergent
    pub(crate) fn resolve_change_set(
        &self,
        set: &RevSet,
        check_immutable: bool,
    ) -> Result<(Vec<Commit>, bool), RevsetError> {
        if set.from.change.hex == set.to.change.hex
            && set.from.change.offset == set.to.change.offset
        {
            let commit = self.resolve_change_id(&set.from)?;
            let is_immutable =
                check_immutable && self.check_immutable([commit.id().clone()]).unwrap_or(false);
            Ok((vec![commit], is_immutable))
        } else {
            let revset_str = format!(
                "{}::{}",
                Self::format_id_str(&set.from),
                Self::format_id_str(&set.to)
            );
            let revset = self.evaluate_revset_str(&revset_str)?;
            let is_immutable =
                check_immutable && self.check_immutable_revset(&*revset).unwrap_or(false);
            let commits = self.resolve_multiple(revset)?;
            Ok((commits, is_immutable))
        }
    }

    // not-really-policy: sometimes we only have a commit, not a change. this is a compromise and will ideally be eliminated
    pub(crate) fn resolve_commit_id(&self, id: &messages::CommitId) -> Result<Commit, RevsetError> {
        let expr = RevsetExpression::commit(
            CommitId::try_from_hex(&id.hex).expect("frontend-validated id"),
        );
        let revset = self.evaluate_revset_expr(self.operation.repo.as_ref(), expr)?;
        self.resolve_single(revset)
    }

    pub(crate) fn resolve_multiple<'op, 'set: 'op, T: AsRef<dyn Revset + 'set>>(
        &'op self,
        revset: T,
    ) -> Result<Vec<Commit>, RevsetError> {
        let store = self.operation.repo.store();
        let commits: Vec<Commit> = revset
            .as_ref()
            .stream()
            .commits(store)
            .try_collect()
            .block_on()
            .map_err(RevsetError::from)?;
        Ok(commits)
    }

    pub(crate) fn resolve_multiple_commits(
        &self,
        ids: &[messages::CommitId],
    ) -> Result<Vec<Commit>, RevsetError> {
        let revset = self.evaluate_revset_commits(ids)?;
        let commits = self.resolve_multiple(revset)?;
        Ok(commits)
    }

    // XXX ideally this would apply the same policy as resolve_single_change
    pub(crate) fn resolve_multiple_changes(
        &self,
        ids: impl IntoIterator<Item = RevId>,
    ) -> Result<Vec<Commit>, RevsetError> {
        let revset =
            self.evaluate_revset_changes(&ids.into_iter().map(|id| id.change).collect_vec())?;
        let commits = self.resolve_multiple(revset)?;
        Ok(commits)
    }

    /*************************************************************
     * Functions for creating temporary per-request derived data *
     *************************************************************/

    // the prefix context caches this itself, but the way it does so is not convenient for us - you need a fallible method and the &dyn Repo
    fn prefix_index(&self) -> IdPrefixIndex<'_> {
        self.operation
            .prefix_context
            .populate(self.repo())
            .expect("prefix context disambiguate_within()")
    }

    fn ref_index(&self) -> &Rc<RefIndex> {
        self.operation
            .ref_index
            .get_or_init(|| Rc::new(build_ref_index(self.operation.repo.as_ref())))
    }

    /// commit id -> workspace name, for workspaces other than the current one
    fn workspace_index(&self) -> &HashMap<CommitId, String> {
        self.operation.workspace_index.get_or_init(|| {
            self.operation
                .repo
                .view()
                .wc_commit_ids()
                .iter()
                .filter(|(_, id)| **id != self.operation.wc_id)
                .map(|(name, id)| (id.clone(), name.as_symbol().to_string()))
                .collect()
        })
    }

    /************************************
     * IPC-message formatting functions *
     ************************************/

    pub(crate) fn format_config(&self) -> Result<messages::RepoConfig> {
        let absolute_path = self.workspace().workspace_root().into();

        let git_remotes = match self.git_repo() {
            Some(repo) => get_git_remote_names(&repo),
            None => vec![],
        };

        let default_revset = config::read_log_revset(
            &self.data.workspace_settings,
            self.env().revset_aliases_map(),
        );

        let mut query_choices = HashMap::new();
        query_choices.insert("default".to_string(), default_revset.clone());

        if self.data.query_choices.is_empty() {
            query_choices.insert(
                "tracked-bookmarks".to_string(),
                "@ | ancestors(bookmarks(), 5)".to_string(),
            );
            query_choices.insert(
                "remote-bookmarks".to_string(),
                "@ | ancestors(remote_bookmarks(), 5)".to_string(),
            );
            query_choices.insert("all-revisions".to_string(), "all()".to_string());
        } else {
            query_choices.extend(self.data.query_choices.clone());
        }

        let latest_query = self
            .session
            .latest_query
            .as_ref()
            .unwrap_or(&default_revset)
            .clone();

        let has_external_diff_tool =
            has_external_tool(&self.data.workspace_settings, "ui.diff-formatter");
        let has_external_merge_tool =
            has_external_tool(&self.data.workspace_settings, "ui.merge-editor");

        Ok(messages::RepoConfig::Workspace {
            absolute_path,
            git_remotes,
            query_choices,
            latest_query,
            status: self.format_status(),
            theme_override: self.data.workspace_settings.ui_theme_override(),
            mark_unpushed_bookmarks: self.data.workspace_settings.ui_mark_unpushed_bookmarks(),
            track_recent_workspaces: self.data.workspace_settings.ui_track_recent_workspaces(),
            ignore_immutable: self.session.ignore_immutable,
            has_external_diff_tool,
            has_external_merge_tool,
        })
    }

    pub(crate) fn format_status(&self) -> messages::RepoStatus {
        messages::RepoStatus {
            operation_description: self
                .operation
                .repo
                .operation()
                .store_operation()
                .metadata
                .description
                .clone(),
            working_copy: self.format_commit_id(&self.operation.wc_id),
        }
    }

    pub(crate) fn format_commit_id(&self, id: &CommitId) -> messages::CommitId {
        let prefix_len = self
            .prefix_index()
            .shortest_commit_prefix_len(self.operation.repo.as_ref(), id)
            .unwrap_or(id.hex().len());

        let hex = id.hex();
        let mut prefix = hex.clone();
        let rest = prefix.split_off(prefix_len);
        messages::CommitId { hex, prefix, rest }
    }

    pub(crate) fn format_change_id(
        &self,
        commit_id: &CommitId,
        change_id: &ChangeId,
    ) -> messages::ChangeId {
        let prefix_len = self
            .prefix_index()
            .shortest_change_prefix_len(self.operation.repo.as_ref(), change_id)
            .block_on()
            .unwrap_or_else(|_| change_id.reverse_hex().len());

        let hex = change_id.reverse_hex();
        let prefix = hex[..prefix_len].to_string();
        let rest = hex[prefix_len..].to_string();

        let targets = self
            .repo()
            .resolve_change_id(change_id)
            .block_on()
            .ok()
            .flatten();

        let offset = targets.as_ref().and_then(|targets| {
            let is_hidden = !targets.has_visible(commit_id);
            let is_divergent = targets.is_divergent();

            if is_hidden || is_divergent {
                targets.find_offset(commit_id)
            } else {
                None
            }
        });

        let is_divergent = targets
            .as_ref()
            .map(|targets| targets.is_divergent())
            .unwrap_or(false);

        messages::ChangeId {
            hex,
            prefix,
            rest,
            offset,
            is_divergent,
        }
    }

    pub(crate) fn format_id(&self, commit: &Commit) -> RevId {
        RevId {
            commit: self.format_commit_id(commit.id()),
            change: self.format_change_id(commit.id(), commit.change_id()),
        }
    }

    pub(crate) fn format_header(
        &self,
        commit: &Commit,
        known_immutable: Option<bool>,
    ) -> Result<RevHeader> {
        let index = self.ref_index();
        let refs = index.get(commit.id()).to_vec();

        let is_immutable = known_immutable
            .map(Result::Ok)
            .unwrap_or_else(|| self.check_immutable(vec![commit.id().clone()]))?;

        Ok(RevHeader {
            id: self.format_id(commit),
            description: commit.description().into(),
            author: commit.author().try_into()?,
            has_conflict: commit.has_conflict(),
            working_copy_of: self.workspace_index().get(commit.id()).cloned(),
            is_working_copy: *commit.id() == self.operation.wc_id
                || self.workspace_index().contains_key(commit.id()),
            is_immutable,
            refs,
            parent_ids: commit
                .parent_ids()
                .iter()
                .map(|commit_id| self.format_commit_id(commit_id))
                .collect(),
        })
    }

    pub(crate) fn format_path<T: AsRef<RepoPath>>(
        &self,
        repo_path: T,
    ) -> Result<messages::TreePath> {
        let base_path = self.workspace().workspace_root();
        let relative_path =
            file_util::relative_path(base_path, &repo_path.as_ref().to_fs_path(base_path)?);
        Ok(messages::TreePath {
            repo_path: repo_path.as_ref().as_internal_file_string().to_owned(),
            relative_path: relative_path.into(),
        })
    }

    fn check_immutable_with_repo<'r>(
        &'r self,
        repo: &'r dyn Repo,
        ids: impl IntoIterator<Item = CommitId>,
    ) -> Result<bool> {
        let check_revset = RevsetExpression::commits(ids.into_iter().collect());
        let immutable_revset = self.env().resolve_immutable_expression(repo, false)?;
        let immutable_revs = check_revset
            .intersection(&immutable_revset)
            .evaluate(repo)?;
        Ok(!immutable_revs.is_empty()?)
    }

    /// checks if any commit in an iterator is immutable
    /// can be slow! this codepath is for use in one-offs where nothing is cached
    pub(crate) fn check_immutable(&self, ids: impl IntoIterator<Item = CommitId>) -> Result<bool> {
        self.check_immutable_with_repo(self.operation.repo.as_ref(), ids)
    }

    /// checks if any commit in a revset is immutable
    /// more efficient than check_immutable thanks to cached membership tests
    pub(crate) fn check_immutable_revset(&self, revset: &dyn Revset) -> Result<bool> {
        let immutable_revset = self.evaluate_immutable()?;
        let contains = immutable_revset.containing_fn();
        let mut stream = revset.stream();
        while let Some(id) = stream.try_next().block_on()? {
            if contains(&id).block_on()? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /*********************************************************************
     * Transaction functions - these are very similar to cli_util        *
     * Ideally in future the code can be extracted to not depend on TUI. *
     *********************************************************************/

    pub(crate) async fn start_transaction(&mut self) -> Result<Transaction> {
        let auto_update_stale = self
            .data
            .workspace_settings
            .get_bool("snapshot.auto-update-stale")
            .unwrap_or(false);
        self.import_and_snapshot(true, auto_update_stale).await?;
        Ok(self.runner.start_transaction())
    }

    /// Finish a transaction, protecting against the working copy landing on an immutable commit.
    /// Most mutations should use this. Use `finish_mutation` only for explicit "edit" actions
    /// where the user has deliberately chosen to edit an immutable commit.
    pub(crate) async fn finish_transaction(
        &mut self,
        tx: Transaction,
        description: impl Into<String>,
    ) -> Result<Option<messages::RepoStatus>> {
        self.finish_transaction_for_edit(tx, description, false)
            .await
    }

    /// Finish a transaction with control over immutability checking.
    /// If `ignore_immutable` is false, and the working copy lands on an immutable commit, a new
    /// commit is created on top to prevent accidental modification. Pass `ignore_immutable: true`
    /// only for explicit "edit" gestures (double-click, Enter, Edit button) where the user has
    /// deliberately chosen to edit an immutable commit.
    pub(crate) async fn finish_transaction_for_edit(
        &mut self,
        tx: Transaction,
        description: impl Into<String>,
        ignore_immutable: bool,
    ) -> Result<Option<messages::RepoStatus>> {
        if !tx.repo().has_changes() {
            return Ok(None);
        }

        self.runner.set_ignore_immutable(ignore_immutable);
        let result = self
            .runner
            .finish_transaction(&LogRunnerEvents, tx, description)
            .await;
        self.runner.set_ignore_immutable(false);
        self.sync_operation();
        result?;

        Ok(Some(self.format_status()))
    }

    /// Imports from Git and snapshots the working copy, unless automatic snapshots are disabled
    /// and `force` is false. Returns whether the repo changed.
    pub(crate) async fn import_and_snapshot(
        &mut self,
        force: bool,
        auto_update_stale: bool,
    ) -> Result<bool> {
        if !(force
            || self
                .data
                .workspace_settings
                .query_auto_snapshot()
                .unwrap_or(!self.is_large))
        {
            return Ok(false);
        }

        let lock = self.runner.lock_git_import_export()?;
        let result = match self
            .runner
            .snapshot(&LogRunnerEvents, &self.snapshot_config, &lock)
            .await
        {
            Err(RunnerError::StaleWorkingCopy(err)) if auto_update_stale => {
                log::info!("Updating stale working copy: {err}");
                self.runner
                    .recover_stale(&LogRunnerEvents, &self.snapshot_config, &lock)
                    .await
            }
            result => result,
        };
        drop(lock);
        let changed = self.sync_operation();
        result?;
        Ok(changed)
    }
}

impl OperationData {
    pub fn new(
        id: &WorkspaceName,
        env: &WorkspaceEnvironment,
        repo: Arc<ReadonlyRepo>,
    ) -> OperationData {
        let wc_id = repo
            .view()
            .get_wc_commit_id(id)
            .expect("No working copy found for workspace")
            .clone();
        let prefix_context = env.new_id_prefix_context();

        OperationData {
            repo,
            wc_id,
            ref_index: OnceCell::default(),
            workspace_index: OnceCell::default(),
            prefix_context,
        }
    }

    fn git_backend(&self) -> Option<&GitBackend> {
        self.repo.store().backend_impl::<GitBackend>()
    }
}

fn absolute_workspace_path(path: &Path) -> Result<PathBuf> {
    dunce::canonicalize(path)
        .with_context(|| format!("Cannot resolve absolute workspace path: {}", path.display()))
}

/// is the repo inside this workspace, rather than a pointer to one elsewhere?
fn hosts_repo(workspace_root: &Path, repo_path: &Path) -> bool {
    let Ok(repo_path) = dunce::canonicalize(repo_path) else {
        return false;
    };

    DefaultWorkspaceLoaderFactory
        .create(workspace_root)
        .ok()
        .and_then(|loader| dunce::canonicalize(loader.repo_path()).ok())
        .is_some_and(|loaded_path| loaded_path == repo_path)
}

/*************************/
/* from commit_templater */
/*************************/

#[derive(Default)]
pub struct RefIndex {
    index: HashMap<CommitId, Vec<messages::StoreRef>>,
}

impl RefIndex {
    fn insert<'a>(
        &mut self,
        ids: impl IntoIterator<Item = &'a CommitId>,
        r#ref: messages::StoreRef,
    ) {
        for id in ids {
            let ref_names = self.index.entry(id.clone()).or_default();
            ref_names.push(r#ref.clone());
        }
    }

    fn get(&self, id: &CommitId) -> &[messages::StoreRef] {
        if let Some(names) = self.index.get(id) {
            names
        } else {
            &[]
        }
    }
}

fn build_ref_index(repo: &ReadonlyRepo) -> RefIndex {
    let potential_remotes = git::get_git_backend(repo.store())
        .ok()
        .map(|git_backend| git_backend.git_repo().remote_names().len())
        .unwrap_or(0);

    let mut index = RefIndex::default();

    for (bookmark_name, bookmark_target) in repo.view().bookmarks() {
        let local_target = bookmark_target.local_target;
        let remote_refs = bookmark_target.remote_refs;
        if local_target.is_present() {
            index.insert(
                local_target.present_adds(),
                messages::StoreRef::LocalBookmark {
                    bookmark_name: bookmark_name.as_str().to_owned(),
                    has_conflict: !local_target.is_resolved(),
                    is_synced: remote_refs.iter().all(|&(remote_name, remote_ref)| {
                        remote_name == REMOTE_NAME_FOR_LOCAL_GIT_REPO
                            || !remote_ref.is_tracked()
                            || remote_ref.target == *local_target
                    }),
                    tracking_remotes: remote_refs
                        .iter()
                        .filter(|&(remote_name, remote_ref)| {
                            *remote_name != REMOTE_NAME_FOR_LOCAL_GIT_REPO
                                && remote_ref.is_tracked()
                        })
                        .map(|&(remote_name, _)| remote_name.as_str().to_owned())
                        .collect(),
                    available_remotes: remote_refs
                        .iter()
                        .filter(|&(remote_name, _)| *remote_name != REMOTE_NAME_FOR_LOCAL_GIT_REPO)
                        .count(),
                    potential_remotes,
                },
            );
        }
        for &(remote_name, remote_ref) in &remote_refs {
            if remote_name == REMOTE_NAME_FOR_LOCAL_GIT_REPO {
                continue;
            }
            index.insert(
                remote_ref.target.present_adds(),
                messages::StoreRef::RemoteBookmark {
                    bookmark_name: bookmark_name.as_str().to_owned(),
                    remote_name: remote_name.as_str().to_owned(),
                    has_conflict: !remote_ref.target.is_resolved(),
                    is_synced: remote_ref.target == *local_target,
                    is_tracked: remote_ref.is_tracked(),
                    is_absent: local_target.is_absent(),
                },
            );
        }
    }

    for (tag_name, tag_target) in repo.view().local_tags() {
        index.insert(
            tag_target.present_adds(),
            messages::StoreRef::Tag {
                tag_name: tag_name.as_str().to_owned(),
            },
        );
    }

    index
}

/// checks whether a jj config key points to an external (non-builtin) tool
fn has_external_tool(settings: &UserSettings, config_key: &'static str) -> bool {
    settings
        .get::<jj_cli::config::CommandNameAndArgs>(config_key)
        .ok()
        .map(|tool| !matches!(tool.as_str(), Some(s) if s.is_empty() || s.starts_with(':')))
        .unwrap_or(false)
}

/************************************************/
/* misc helpers that should be better organised */
/************************************************/

fn runner_options() -> RunnerOptions {
    RunnerOptions {
        ignore_working_copy: false,
        integrate_operation: true,
        ignore_immutable: false,
        merge_concurrent_operations: true,
        transaction_attributes: vec![],
    }
}

fn new_workspace_environment(
    workspace: &Workspace,
    aliases_map: RevsetAliasesMap,
    fileset_aliases_map: FilesetAliasesMap,
) -> Result<Arc<WorkspaceEnvironment>> {
    let mut diagnostics = WorkspaceEnvironmentDiagnostics::new();
    let env = WorkspaceEnvironment::new(
        &mut diagnostics,
        workspace,
        workspace.workspace_root(),
        Arc::new(RevsetExtensions::default()),
        fileset_aliases_map,
        aliases_map,
    )?;
    log::debug!("{diagnostics:?}");
    Ok(Arc::new(env))
}

fn load_snapshot_config(
    workspace: &Workspace,
    env: &WorkspaceEnvironment,
) -> Result<SnapshotConfig> {
    let mut diagnostics = SnapshotConfigDiagnostics::new();
    let home_dir = etcetera::home_dir().ok();
    let config = SnapshotConfig::load(
        &mut diagnostics,
        env,
        workspace.repo_loader().store(),
        home_dir.as_deref(),
    )?;
    log::debug!("{diagnostics:?}");
    Ok(config)
}

/// Logs the events reported by the runner, since there's no console to print them on.
struct LogRunnerEvents;

impl RunnerEvents for LogRunnerEvents {
    fn event(
        &self,
        event: RunnerEvent<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        log::info!("{event:?}");
        Ok(())
    }
}
