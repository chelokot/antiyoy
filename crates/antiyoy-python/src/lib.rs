#![forbid(unsafe_code)]

use antiyoy_agents::{Agent, GreedyAgent, PuctConfig, PuctSearch, SearchAgent, SearchConfig};
use antiyoy_core::{
    EconomyMetric, GeneratorConfig, Objective, PlayerId, Relation, Rules, VictoryCondition,
};
use antiyoy_rl::{BatchEnv, BatchObservation, StepResult, encoded_rule_features};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rayon::prelude::*;

#[pyfunction]
fn encode_rule_features<'py>(
    py: Python<'py>,
    serialized: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let rules: Rules = serde_json::from_str(serialized)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray1::from_slice(py, &encoded_rule_features(&rules)))
}

#[pyclass(module = "antiyoy_rl._native", frozen)]
struct ProceduralConfig {
    inner: GeneratorConfig,
}

#[pymethods]
impl ProceduralConfig {
    #[new]
    #[pyo3(signature = (width=31, height=21, players=2, seed=1, land_density_per_million=650_000, starting_province_size=5, starting_money=10, tree_density_per_million=150_000, neutral_tower_density_per_million=20_000, neutral_capital_density_per_million=10_000, grave_density_per_million=15_000))]
    #[expect(clippy::too_many_arguments)]
    fn new(
        width: u16,
        height: u16,
        players: u8,
        seed: u64,
        land_density_per_million: u32,
        starting_province_size: u16,
        starting_money: i64,
        tree_density_per_million: u32,
        neutral_tower_density_per_million: u32,
        neutral_capital_density_per_million: u32,
        grave_density_per_million: u32,
    ) -> Self {
        Self {
            inner: GeneratorConfig {
                schema_version: antiyoy_core::GENERATOR_SCHEMA_VERSION,
                width,
                height,
                players,
                seed,
                land_density_per_million,
                starting_province_size,
                starting_money,
                tree_density_per_million,
                neutral_tower_density_per_million,
                neutral_capital_density_per_million,
                grave_density_per_million,
            },
        }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(runtime_error)
    }
}

#[pyclass(module = "antiyoy_rl._native", frozen)]
struct ScenarioObjective {
    inner: Objective,
}

#[pyclass(module = "antiyoy_rl._native")]
struct PolicySearchBatch {
    searches: Vec<Option<PuctSearch>>,
    observation: BatchObservation,
    pending_leaves: Vec<(usize, u64)>,
    fog: bool,
}

#[pymethods]
impl PolicySearchBatch {
    fn is_complete(&self) -> bool {
        self.searches.iter().flatten().all(PuctSearch::is_complete)
    }

    fn select_leaves<'py>(
        &mut self,
        py: Python<'py>,
        maximum_batch_size: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if maximum_batch_size == 0 {
            return Err(PyValueError::new_err(
                "PUCT leaf batch size must be positive",
            ));
        }
        if !self.pending_leaves.is_empty() {
            return Err(PyRuntimeError::new_err(
                "complete the pending PUCT leaf batch before selecting another",
            ));
        }
        while self.pending_leaves.len() < maximum_batch_size && !self.is_complete() {
            let mut progressed = false;
            for (environment, search) in self.searches.iter_mut().enumerate() {
                let Some(search) = search else {
                    continue;
                };
                let before = search.stats();
                if let Some(leaf) = search.select_leaves(1).into_iter().next() {
                    self.pending_leaves.push((environment, leaf.token));
                    progressed = true;
                } else {
                    progressed |= search.stats() != before;
                }
                if self.pending_leaves.len() == maximum_batch_size {
                    break;
                }
            }
            if !progressed {
                break;
            }
        }
        if self.pending_leaves.is_empty() && !self.is_complete() {
            return Err(PyRuntimeError::new_err(
                "PUCT search stalled without an evaluable leaf",
            ));
        }
        let games = self
            .pending_leaves
            .iter()
            .map(|(environment, token)| {
                self.searches[*environment]
                    .as_ref()
                    .and_then(|search| search.leaf(*token))
                    .ok_or_else(|| PyRuntimeError::new_err("PUCT leaf disappeared"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.observation.observe_games(&games, self.fog);
        let dictionary = observation_dict(py, &self.observation)?;
        dictionary.set_item(
            "search_environments",
            PyArray1::from_vec(
                py,
                self.pending_leaves
                    .iter()
                    .map(|(environment, _)| *environment as u64)
                    .collect(),
            ),
        )?;
        dictionary.set_item(
            "search_tokens",
            PyArray1::from_vec(
                py,
                self.pending_leaves
                    .iter()
                    .map(|(_, token)| *token)
                    .collect(),
            ),
        )?;
        Ok(dictionary)
    }

    #[expect(clippy::needless_pass_by_value)]
    fn complete_leaves(
        &mut self,
        priors: PyReadonlyArray1<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let priors = priors
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let values = values
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if priors.len() != self.observation.actions.len() {
            return Err(PyValueError::new_err(format!(
                "PUCT prior vector has length {}, expected {}",
                priors.len(),
                self.observation.actions.len()
            )));
        }
        if values.len() != self.pending_leaves.len() {
            return Err(PyValueError::new_err(format!(
                "PUCT value vector has length {}, expected {}",
                values.len(),
                self.pending_leaves.len()
            )));
        }
        for (leaf, boundaries) in self
            .pending_leaves
            .iter()
            .enumerate()
            .map(|(leaf, _)| (leaf, &self.observation.action_offsets[leaf..=leaf + 1]))
        {
            let leaf_priors = &priors[boundaries[0]..boundaries[1]];
            let mass = leaf_priors.iter().copied().sum::<f32>();
            if !values[leaf].is_finite()
                || !mass.is_finite()
                || mass <= 0.0
                || leaf_priors
                    .iter()
                    .any(|prior| !prior.is_finite() || *prior < 0.0)
            {
                return Err(PyValueError::new_err(
                    "PUCT priors and values must be finite with positive prior mass",
                ));
            }
        }
        for (leaf, (environment, token)) in self.pending_leaves.iter().copied().enumerate() {
            let start = self.observation.action_offsets[leaf];
            let end = self.observation.action_offsets[leaf + 1];
            let leaf_priors = priors[start..end]
                .iter()
                .map(|prior| f64::from(*prior))
                .collect::<Vec<_>>();
            self.searches[environment]
                .as_mut()
                .expect("pending PUCT leaves belong to active searches")
                .complete_leaf(token, &leaf_priors, f64::from(values[leaf]))
                .map_err(runtime_error)?;
        }
        self.pending_leaves.clear();
        Ok(())
    }

    fn action_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
        if !self.is_complete() || !self.pending_leaves.is_empty() {
            return Err(PyRuntimeError::new_err("PUCT search is not complete"));
        }
        let indices = self
            .searches
            .iter()
            .map(|search| {
                search.as_ref().map_or(Ok(0), |search| {
                    u64::try_from(search.selected_action_index().map_err(runtime_error)?)
                        .map_err(|_| PyRuntimeError::new_err("action index does not fit u64"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyArray1::from_vec(py, indices))
    }

    fn root_targets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.is_complete() || !self.pending_leaves.is_empty() {
            return Err(PyRuntimeError::new_err("PUCT search is not complete"));
        }
        let mut offsets = Vec::with_capacity(self.searches.len() + 1);
        let mut probabilities = Vec::new();
        offsets.push(0_u64);
        for search in &self.searches {
            if let Some(search) = search {
                probabilities.extend(search.root_target_probabilities().map_err(runtime_error)?);
            }
            offsets.push(
                u64::try_from(probabilities.len())
                    .map_err(|_| PyRuntimeError::new_err("root target offset does not fit u64"))?,
            );
        }
        let dictionary = PyDict::new(py);
        dictionary.set_item("offsets", PyArray1::from_vec(py, offsets))?;
        dictionary.set_item("probabilities", PyArray1::from_vec(py, probabilities))?;
        Ok(dictionary)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self
            .searches
            .iter()
            .map(|search| search.as_ref().map(PuctSearch::stats).unwrap_or_default())
            .collect::<Vec<_>>();
        let dictionary = PyDict::new(py);
        dictionary.set_item(
            "nodes",
            PyArray1::from_vec(py, stats.iter().map(|stats| stats.nodes as u64).collect()),
        )?;
        dictionary.set_item(
            "completed_simulations",
            PyArray1::from_vec(
                py,
                stats
                    .iter()
                    .map(|stats| stats.completed_simulations)
                    .collect(),
            ),
        )?;
        dictionary.set_item(
            "maximum_depth",
            PyArray1::from_vec(
                py,
                stats
                    .iter()
                    .map(|stats| stats.maximum_depth as u64)
                    .collect(),
            ),
        )?;
        dictionary.set_item(
            "root_visits",
            PyArray1::from_vec(py, stats.iter().map(|stats| stats.root_visits).collect()),
        )?;
        Ok(dictionary)
    }
}

#[pymethods]
impl ScenarioObjective {
    #[staticmethod]
    fn from_json(serialized: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(serialized)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn domination() -> Self {
        Self {
            inner: Objective::default(),
        }
    }

    #[staticmethod]
    fn diplomatic_victory(player: u8) -> Self {
        Self::new(VictoryCondition::DiplomaticVictory {
            player: PlayerId(player),
        })
    }

    #[staticmethod]
    fn survive_through_round(player: u8, round: u32) -> Self {
        Self::new(VictoryCondition::SurviveThroughRound {
            player: PlayerId(player),
            round,
        })
    }

    #[staticmethod]
    fn destroy_player(player: u8, target: u8) -> Self {
        Self::new(VictoryCondition::DestroyPlayer {
            player: PlayerId(player),
            target: PlayerId(target),
        })
    }

    #[staticmethod]
    fn reach_economy(player: u8, metric: &str, minimum: i64) -> PyResult<Self> {
        let metric = match metric {
            "gross_income" => EconomyMetric::GrossIncome,
            "profit" => EconomyMetric::Profit,
            "treasury" => EconomyMetric::Treasury,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown economy objective metric: {metric}"
                )));
            }
        };
        Ok(Self::new(VictoryCondition::ReachEconomy {
            player: PlayerId(player),
            metric,
            minimum,
        }))
    }

    #[staticmethod]
    fn ensure_player_victory(player: u8) -> Self {
        Self::new(VictoryCondition::EnsurePlayerVictory {
            player: PlayerId(player),
        })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(runtime_error)
    }
}

impl ScenarioObjective {
    fn new(condition: VictoryCondition) -> Self {
        Self {
            inner: Objective {
                schema_version: antiyoy_core::OBJECTIVE_SCHEMA_VERSION,
                condition,
            },
        }
    }
}

#[pyclass(module = "antiyoy_rl._native")]
struct VectorEnv {
    batch: BatchEnv,
    observation: BatchObservation,
    search_config: Option<SearchConfig>,
    search_agents: Vec<SearchAgent>,
}

#[pymethods]
impl VectorEnv {
    #[new]
    #[pyo3(signature = (environments, width=11, height=9, seed=1, action_limit=1000, profile="classic_generic_2022", fog=false, diplomacy=false, initial_relation="neutral", objective=None))]
    #[expect(clippy::needless_pass_by_value)]
    #[expect(clippy::too_many_arguments)]
    fn new(
        environments: usize,
        width: u16,
        height: u16,
        seed: u64,
        action_limit: u32,
        profile: &str,
        fog: bool,
        diplomacy: bool,
        initial_relation: &str,
        objective: Option<PyRef<'_, ScenarioObjective>>,
    ) -> PyResult<Self> {
        let mut rules = rules_for_profile(profile)?;
        configure_diplomacy(&mut rules, diplomacy, initial_relation)?;
        let mut batch =
            BatchEnv::symmetric_duels(rules, environments, width, height, seed, action_limit)
                .map_err(runtime_error)?;
        apply_objective(&mut batch, objective.as_deref().map(|value| &value.inner))?;
        batch.set_fog(fog);
        Ok(Self::from_batch(batch))
    }

    #[staticmethod]
    #[pyo3(signature = (profiles, width=11, height=9, seed=1, action_limit=1000, fog=false, diplomacy=false, initial_relation="neutral", objective=None))]
    #[expect(clippy::needless_pass_by_value)]
    #[expect(clippy::too_many_arguments)]
    fn mixed(
        profiles: Vec<String>,
        width: u16,
        height: u16,
        seed: u64,
        action_limit: u32,
        fog: bool,
        diplomacy: bool,
        initial_relation: &str,
        objective: Option<PyRef<'_, ScenarioObjective>>,
    ) -> PyResult<Self> {
        let rules = profiles
            .iter()
            .map(|profile| {
                let mut rules = rules_for_profile(profile)?;
                configure_diplomacy(&mut rules, diplomacy, initial_relation)?;
                Ok(rules)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut batch = BatchEnv::symmetric_duels_mixed(rules, width, height, seed, action_limit)
            .map_err(runtime_error)?;
        apply_objective(&mut batch, objective.as_deref().map(|value| &value.inner))?;
        batch.set_fog(fog);
        Ok(Self::from_batch(batch))
    }

    #[staticmethod]
    #[pyo3(signature = (environments, config, action_limit=1000, profile="classic_generic_2022", fog=false, diplomacy=false, initial_relation="neutral", objective=None))]
    #[expect(clippy::needless_pass_by_value)]
    #[expect(clippy::too_many_arguments)]
    fn procedural(
        environments: usize,
        config: PyRef<'_, ProceduralConfig>,
        action_limit: u32,
        profile: &str,
        fog: bool,
        diplomacy: bool,
        initial_relation: &str,
        objective: Option<PyRef<'_, ScenarioObjective>>,
    ) -> PyResult<Self> {
        let mut rules = rules_for_profile(profile)?;
        configure_diplomacy(&mut rules, diplomacy, initial_relation)?;
        let mut batch = BatchEnv::procedural(rules, environments, &config.inner, action_limit)
            .map_err(runtime_error)?;
        apply_objective(&mut batch, objective.as_deref().map(|value| &value.inner))?;
        batch.set_fog(fog);
        Ok(Self::from_batch(batch))
    }

    #[staticmethod]
    #[pyo3(signature = (profiles, config, action_limit=1000, fog=false, diplomacy=false, initial_relation="neutral", objective=None))]
    #[expect(clippy::needless_pass_by_value)]
    fn procedural_mixed(
        profiles: Vec<String>,
        config: PyRef<'_, ProceduralConfig>,
        action_limit: u32,
        fog: bool,
        diplomacy: bool,
        initial_relation: &str,
        objective: Option<PyRef<'_, ScenarioObjective>>,
    ) -> PyResult<Self> {
        let rules = profiles
            .iter()
            .map(|profile| {
                let mut rules = rules_for_profile(profile)?;
                configure_diplomacy(&mut rules, diplomacy, initial_relation)?;
                Ok(rules)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut batch = BatchEnv::procedural_mixed(rules, &config.inner, action_limit)
            .map_err(runtime_error)?;
        apply_objective(&mut batch, objective.as_deref().map(|value| &value.inner))?;
        batch.set_fog(fog);
        Ok(Self::from_batch(batch))
    }

    #[staticmethod]
    #[pyo3(signature = (profiles, configs, action_limit=1000, fog=false, diplomacy=false, initial_relation="neutral", objective=None))]
    #[expect(clippy::needless_pass_by_value)]
    #[expect(clippy::too_many_arguments)]
    fn procedural_domains(
        py: Python<'_>,
        profiles: Vec<String>,
        configs: Vec<Py<ProceduralConfig>>,
        action_limit: u32,
        fog: bool,
        diplomacy: bool,
        initial_relation: &str,
        objective: Option<PyRef<'_, ScenarioObjective>>,
    ) -> PyResult<Self> {
        let rules = profiles
            .iter()
            .map(|profile| {
                let mut rules = rules_for_profile(profile)?;
                configure_diplomacy(&mut rules, diplomacy, initial_relation)?;
                Ok(rules)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let generators = configs
            .iter()
            .map(|config| config.borrow(py).inner.clone())
            .collect();
        let mut batch =
            BatchEnv::procedural_domains(rules, generators, action_limit).map_err(runtime_error)?;
        apply_objective(&mut batch, objective.as_deref().map(|value| &value.inner))?;
        batch.set_fog(fog);
        Ok(Self::from_batch(batch))
    }

    #[getter]
    fn environments(&self) -> usize {
        self.batch.len()
    }

    fn observe<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.batch.observe(&mut self.observation);
        observation_dict(py, &self.observation)
    }

    #[expect(clippy::needless_pass_by_value)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action_indices: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let indices = action_indices
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| PyValueError::new_err("action index does not fit usize"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let results = py
            .detach(|| self.batch.step_all(&indices))
            .map_err(runtime_error)?;
        step_dict(py, &results)
    }

    fn reset(&mut self, environment: usize, seed: u64) -> PyResult<()> {
        self.batch
            .reset_with_seed(environment, seed)
            .map_err(runtime_error)
    }

    fn done(&self) -> Vec<bool> {
        (0..self.batch.len())
            .map(|index| self.batch.is_done(index).unwrap_or(true))
            .collect()
    }

    fn rules_json(&self) -> PyResult<String> {
        let game = self
            .batch
            .game(0)
            .ok_or_else(|| PyRuntimeError::new_err("environment batch is empty"))?;
        serde_json::to_string(game.rules()).map_err(runtime_error)
    }

    fn rules_jsons(&self) -> PyResult<Vec<String>> {
        (0..self.batch.len())
            .map(|index| {
                let game = self
                    .batch
                    .game(index)
                    .ok_or_else(|| PyRuntimeError::new_err("environment index disappeared"))?;
                serde_json::to_string(game.rules()).map_err(runtime_error)
            })
            .collect()
    }

    fn generator_jsons(&self) -> PyResult<Vec<Option<String>>> {
        (0..self.batch.len())
            .map(|index| {
                self.batch
                    .generator_config(index)
                    .map(serde_json::to_string)
                    .transpose()
                    .map_err(runtime_error)
            })
            .collect()
    }

    fn objective_jsons(&self) -> PyResult<Vec<String>> {
        (0..self.batch.len())
            .map(|index| {
                let objective = self
                    .batch
                    .objective(index)
                    .ok_or_else(|| PyRuntimeError::new_err("environment index disappeared"))?;
                serde_json::to_string(objective).map_err(runtime_error)
            })
            .collect()
    }

    fn greedy_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let indices = py.detach(|| {
            (0..self.batch.len())
                .into_par_iter()
                .map(|index| {
                    let game = self
                        .batch
                        .game(index)
                        .ok_or_else(|| PyRuntimeError::new_err("environment index disappeared"))?;
                    let actions = self
                        .batch
                        .legal_actions(index)
                        .ok_or_else(|| PyRuntimeError::new_err("legal action index disappeared"))?;
                    let mut agent = GreedyAgent::new("greedy");
                    let selected = agent.select_action(game, actions);
                    let position = actions
                        .iter()
                        .position(|action| *action == selected)
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("greedy action is not in the legal action list")
                        })?;
                    u64::try_from(position)
                        .map_err(|_| PyRuntimeError::new_err("action index does not fit u64"))
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        Ok(PyArray1::from_vec(py, indices))
    }

    #[pyo3(signature = (node_budget=256, exploration=1.5, virtual_loss=1.0, maximum_depth=128, root_value_weight=None, active_mask=None))]
    fn policy_search(
        &self,
        node_budget: usize,
        exploration: f64,
        virtual_loss: f64,
        maximum_depth: usize,
        root_value_weight: Option<f64>,
        active_mask: Option<PyReadonlyArray1<'_, u8>>,
    ) -> PyResult<PolicySearchBatch> {
        let active = active_mask_values(active_mask, self.batch.len())?;
        let config = PuctConfig {
            node_budget,
            exploration,
            virtual_loss,
            maximum_depth,
            root_value_weight,
        };
        let searches = (0..self.batch.len())
            .map(|index| {
                if !active[index] {
                    return Ok(None);
                }
                let game = self
                    .batch
                    .game(index)
                    .ok_or_else(|| PyRuntimeError::new_err("environment index disappeared"))?;
                let actions = self
                    .batch
                    .legal_actions(index)
                    .ok_or_else(|| PyRuntimeError::new_err("legal action index disappeared"))?;
                PuctSearch::new(game, actions, config)
                    .map(Some)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PolicySearchBatch {
            searches,
            observation: BatchObservation::default(),
            pending_leaves: Vec::new(),
            fog: self.batch.fog_enabled(),
        })
    }

    #[pyo3(signature = (node_budget=2048, beam_width=32, branch_width=48, maximum_actions_per_turn=24, active_mask=None))]
    fn search_actions<'py>(
        &mut self,
        py: Python<'py>,
        node_budget: usize,
        beam_width: usize,
        branch_width: usize,
        maximum_actions_per_turn: usize,
        active_mask: Option<PyReadonlyArray1<'py, u8>>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let config = SearchConfig {
            node_budget,
            beam_width,
            branch_width,
            maximum_actions_per_turn,
        };
        let active = active_mask_values(active_mask, self.batch.len())?;
        self.select_search_actions(py, config, Some(&active), true)
    }

    #[pyo3(signature = (node_budget=2048, beam_width=32, branch_width=48, maximum_actions_per_turn=24))]
    fn search_actions_replanned<'py>(
        &mut self,
        py: Python<'py>,
        node_budget: usize,
        beam_width: usize,
        branch_width: usize,
        maximum_actions_per_turn: usize,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        self.select_search_actions(
            py,
            SearchConfig {
                node_budget,
                beam_width,
                branch_width,
                maximum_actions_per_turn,
            },
            None,
            false,
        )
    }

    fn search_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_vec(
            py,
            self.search_agents
                .iter()
                .map(SearchAgent::search_count)
                .collect(),
        )
    }
}

impl VectorEnv {
    fn select_search_actions<'py>(
        &mut self,
        py: Python<'py>,
        config: SearchConfig,
        active: Option<&[bool]>,
        reuse_plan: bool,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        if self.search_config != Some(config) {
            self.search_agents = (0..self.batch.len())
                .map(|index| SearchAgent::with_config(format!("search-{index}"), config))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            self.search_config = Some(config);
        }
        let batch = &self.batch;
        let indices = py.detach(|| {
            self.search_agents
                .par_iter_mut()
                .enumerate()
                .map(|(index, agent)| {
                    if active.as_ref().is_some_and(|mask| !mask[index]) {
                        return Ok(0);
                    }
                    let game = batch
                        .game(index)
                        .ok_or_else(|| PyRuntimeError::new_err("environment index disappeared"))?;
                    let actions = batch
                        .legal_actions(index)
                        .ok_or_else(|| PyRuntimeError::new_err("legal action index disappeared"))?;
                    if actions.is_empty() {
                        return Err(PyRuntimeError::new_err(format!(
                            "environment {index} is done and must be reset"
                        )));
                    }
                    if !reuse_plan {
                        agent.clear_plan();
                    }
                    let selected = agent.select_action(game, actions);
                    let position = actions
                        .iter()
                        .position(|action| *action == selected)
                        .expect("search agent must return a legal action");
                    u64::try_from(position)
                        .map_err(|_| PyRuntimeError::new_err("action index does not fit u64"))
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        Ok(PyArray1::from_vec(py, indices))
    }

    fn from_batch(batch: BatchEnv) -> Self {
        Self {
            batch,
            observation: BatchObservation::default(),
            search_config: None,
            search_agents: Vec::new(),
        }
    }
}

fn rules_for_profile(profile: &str) -> PyResult<Rules> {
    match profile {
        "classic_generic_2022" => Ok(Rules::classic_generic()),
        "classic_slay_2022" => Ok(Rules::classic_slay()),
        "online_default_v1" => Ok(Rules::online_default_v1()),
        "online_classic_v1" => Ok(Rules::online_classic_v1()),
        "online_duel_v1" => Ok(Rules::online_duel_v1()),
        "online_experimental_v1" => Ok(Rules::online_experimental_v1()),
        "online_experimental_v2_260801" => Ok(Rules::online_experimental_v2_260801()),
        _ => Err(PyValueError::new_err(format!(
            "unknown rules profile: {profile}"
        ))),
    }
}

fn configure_diplomacy(rules: &mut Rules, enabled: bool, initial_relation: &str) -> PyResult<()> {
    rules.diplomacy.enabled = enabled;
    rules.diplomacy.initial_relation = match initial_relation {
        "war" => Relation::War,
        "neutral" => Relation::Neutral,
        "friend" => Relation::Friend,
        "alliance" => Relation::Alliance,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown initial relation: {initial_relation}"
            )));
        }
    };
    Ok(())
}

fn apply_objective(batch: &mut BatchEnv, objective: Option<&Objective>) -> PyResult<()> {
    let Some(objective) = objective else {
        return Ok(());
    };
    for index in 0..batch.len() {
        batch
            .set_objective(index, objective.clone())
            .map_err(runtime_error)?;
    }
    Ok(())
}

fn observation_dict<'py>(
    py: Python<'py>,
    observation: &BatchObservation,
) -> PyResult<Bound<'py, PyDict>> {
    let dictionary = PyDict::new(py);
    dictionary.set_item("version", observation.version)?;
    dictionary.set_item(
        "cell_offsets",
        PyArray1::from_vec(py, offsets(&observation.cell_offsets)?),
    )?;
    dictionary.set_item(
        "province_offsets",
        PyArray1::from_vec(py, offsets(&observation.province_offsets)?),
    )?;
    dictionary.set_item(
        "action_offsets",
        PyArray1::from_vec(py, offsets(&observation.action_offsets)?),
    )?;
    dictionary.set_item(
        "relation_offsets",
        PyArray1::from_vec(py, offsets(&observation.relation_offsets)?),
    )?;
    dictionary.set_item("widths", PyArray1::from_slice(py, &observation.widths))?;
    dictionary.set_item("heights", PyArray1::from_slice(py, &observation.heights))?;
    dictionary.set_item(
        "active_players",
        PyArray1::from_slice(py, &observation.active_players),
    )?;
    dictionary.set_item(
        "player_counts",
        PyArray1::from_slice(py, &observation.player_counts),
    )?;
    dictionary.set_item("rounds", PyArray1::from_slice(py, &observation.rounds))?;
    dictionary.set_item("playable", PyArray1::from_slice(py, &observation.playable))?;
    dictionary.set_item("visible", PyArray1::from_slice(py, &observation.visible))?;
    dictionary.set_item("owners", PyArray1::from_slice(py, &observation.owners))?;
    dictionary.set_item("objects", PyArray1::from_slice(py, &observation.objects))?;
    dictionary.set_item(
        "unit_strengths",
        PyArray1::from_slice(py, &observation.unit_strengths),
    )?;
    dictionary.set_item("ready", PyArray1::from_slice(py, &observation.ready))?;
    dictionary.set_item("defenses", PyArray1::from_slice(py, &observation.defenses))?;
    dictionary.set_item(
        "province_ids",
        PyArray1::from_slice(py, &observation.province_ids),
    )?;
    dictionary.set_item(
        "province_owners",
        PyArray1::from_slice(py, &observation.province_owners),
    )?;
    dictionary.set_item(
        "province_money",
        PyArray1::from_slice(py, &observation.province_money),
    )?;
    dictionary.set_item(
        "province_profit",
        PyArray1::from_slice(py, &observation.province_profit),
    )?;
    dictionary.set_item(
        "province_capitals",
        PyArray1::from_slice(py, &observation.province_capitals),
    )?;
    dictionary.set_item(
        "province_sizes",
        PyArray1::from_vec(py, offsets(&observation.province_sizes)?),
    )?;
    add_action_arrays(py, &dictionary, observation)?;
    dictionary.set_item(
        "relations",
        PyArray1::from_slice(py, &observation.relations),
    )?;
    dictionary.set_item(
        "proposals",
        PyArray1::from_slice(py, &observation.proposals),
    )?;
    Ok(dictionary)
}

fn add_action_arrays<'py>(
    py: Python<'py>,
    dictionary: &Bound<'py, PyDict>,
    observation: &BatchObservation,
) -> PyResult<()> {
    dictionary.set_item(
        "action_kinds",
        PyArray1::from_vec(
            py,
            observation
                .actions
                .iter()
                .map(|action| action.kind.code())
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "action_sources",
        PyArray1::from_vec(
            py,
            observation
                .actions
                .iter()
                .map(|action| action.source)
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "action_targets",
        PyArray1::from_vec(
            py,
            observation
                .actions
                .iter()
                .map(|action| action.target)
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "action_parameters",
        PyArray1::from_vec(
            py,
            observation
                .actions
                .iter()
                .map(|action| action.parameter)
                .collect(),
        ),
    )?;
    Ok(())
}

fn step_dict<'py>(py: Python<'py>, results: &[StepResult]) -> PyResult<Bound<'py, PyDict>> {
    let dictionary = PyDict::new(py);
    dictionary.set_item(
        "actors",
        PyArray1::from_vec(py, results.iter().map(|result| result.actor.0).collect()),
    )?;
    dictionary.set_item(
        "outcomes",
        PyArray1::from_vec(
            py,
            results.iter().map(|result| result.reward.outcome).collect(),
        ),
    )?;
    dictionary.set_item(
        "territory_delta",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| result.reward.territory_delta)
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "treasury_delta",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| result.reward.treasury_delta)
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "unit_strength_delta",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| result.reward.unit_strength_delta)
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "rounds",
        PyArray1::from_vec(py, results.iter().map(|result| result.round).collect()),
    )?;
    dictionary.set_item(
        "terminal",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| u8::from(result.terminal))
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "truncated",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| u8::from(result.truncated))
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "objective_satisfied",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| u8::from(result.objective_satisfied))
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "winners",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| result.winner.map_or(u8::MAX, |winner| winner.0))
                .collect(),
        ),
    )?;
    dictionary.set_item(
        "adjudicated_winners",
        PyArray1::from_vec(
            py,
            results
                .iter()
                .map(|result| result.adjudicated_winner.map_or(u8::MAX, |winner| winner.0))
                .collect(),
        ),
    )?;
    Ok(dictionary)
}

fn offsets(values: &[usize]) -> PyResult<Vec<u64>> {
    values
        .iter()
        .copied()
        .map(|value| {
            u64::try_from(value).map_err(|_| PyRuntimeError::new_err("offset does not fit u64"))
        })
        .collect()
}

fn active_mask_values(
    mask: Option<PyReadonlyArray1<'_, u8>>,
    environments: usize,
) -> PyResult<Vec<bool>> {
    let Some(mask) = mask else {
        return Ok(vec![true; environments]);
    };
    let values = mask
        .as_slice()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if values.len() != environments {
        return Err(PyValueError::new_err(format!(
            "active mask has length {}, expected {environments}",
            values.len()
        )));
    }
    Ok(values.iter().map(|value| *value != 0).collect())
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode_rule_features, module)?)?;
    module.add_class::<ProceduralConfig>()?;
    module.add_class::<ScenarioObjective>()?;
    module.add_class::<PolicySearchBatch>()?;
    module.add_class::<VectorEnv>()?;
    module.add("OBSERVATION_VERSION", antiyoy_rl::OBSERVATION_VERSION)?;
    module.add(
        "GENERATOR_SCHEMA_VERSION",
        antiyoy_core::GENERATOR_SCHEMA_VERSION,
    )?;
    module.add(
        "OBJECTIVE_SCHEMA_VERSION",
        antiyoy_core::OBJECTIVE_SCHEMA_VERSION,
    )?;
    Ok(())
}
