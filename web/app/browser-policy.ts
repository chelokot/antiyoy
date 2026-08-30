import type { InferenceSession, Tensor as OrtTensor } from "onnxruntime-web";

type PolicyActionKind =
  | "EndTurn"
  | "Move"
  | "Recruit"
  | "Build"
  | "PlantTree"
  | "Diplomacy";

type PolicyAction = {
  kind: PolicyActionKind;
  source: number;
  target: number;
  parameter: number;
};

type PolicyObservation = {
  widths: number[];
  heights: number[];
  active_players: number[];
  player_counts: number[];
  rounds: number[];
  playable: number[];
  visible: number[];
  owners: number[];
  objects: number[];
  unit_strengths: number[];
  ready: number[];
  defenses: number[];
  province_ids: number[];
  province_money: number[];
  province_profit: number[];
  province_sizes: number[];
  rule_features: number[];
  actions: PolicyAction[];
};

export type PolicyDecision = {
  actionIndex: number;
  milliseconds: number;
  legalActions: number;
};

const ACTION_KIND_CODES: Record<PolicyActionKind, number> = {
  EndTurn: 0,
  Move: 1,
  Recruit: 2,
  Build: 3,
  PlantTree: 4,
  Diplomacy: 5,
};

function int64Tensor(
  Tensor: typeof import("onnxruntime-web").Tensor,
  values: number[],
): OrtTensor {
  return new Tensor("int64", BigInt64Array.from(values, BigInt), [values.length]);
}

function provinceFeatures(observation: PolicyObservation): Float32Array {
  const features = new Float32Array(observation.province_ids.length * 3);
  for (let cell = 0; cell < observation.province_ids.length; cell += 1) {
    const province = observation.province_ids[cell];
    if (province === 65_535) {
      continue;
    }
    const values = [
      observation.province_money[province],
      observation.province_profit[province],
      observation.province_sizes[province],
    ];
    for (let feature = 0; feature < values.length; feature += 1) {
      const value = values[feature];
      features[cell * 3 + feature] = Math.sign(value) * Math.log1p(Math.abs(value));
    }
  }
  return features;
}

export class RoutedBrowserPolicy {
  private constructor(
    private readonly session: InferenceSession,
    private readonly Tensor: typeof import("onnxruntime-web").Tensor,
  ) {}

  static async load(
    modelUrl: string,
    runtimeBaseUrl = `${window.location.origin}/`,
  ): Promise<RoutedBrowserPolicy> {
    const runtime = await import("onnxruntime-web/wasm");
    runtime.env.wasm.numThreads = 1;
    runtime.env.wasm.wasmPaths = runtimeBaseUrl;
    const session = await runtime.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new RoutedBrowserPolicy(session, runtime.Tensor);
  }

  async decide(serializedObservation: string): Promise<PolicyDecision> {
    const observation = JSON.parse(serializedObservation) as PolicyObservation;
    if (
      observation.widths.length !== 1
      || observation.heights.length !== 1
      || observation.widths[0] !== 11
      || observation.heights[0] !== 9
      || observation.rule_features.length !== 45
    ) {
      throw new Error("Neural policy requires one 11×9 environment with 45 rule features");
    }
    const actions = observation.actions;
    const inputs: Record<string, OrtTensor> = {
      playable: new this.Tensor(
        "float32",
        Float32Array.from(observation.playable),
        [observation.playable.length],
      ),
      visible: int64Tensor(this.Tensor, observation.visible),
      owners: int64Tensor(this.Tensor, observation.owners),
      objects: int64Tensor(this.Tensor, observation.objects),
      unit_strengths: int64Tensor(this.Tensor, observation.unit_strengths),
      ready: int64Tensor(this.Tensor, observation.ready),
      defenses: int64Tensor(this.Tensor, observation.defenses),
      province_features: new this.Tensor(
        "float32",
        provinceFeatures(observation),
        [observation.province_ids.length, 3],
      ),
      rule_features: new this.Tensor(
        "float32",
        Float32Array.from(observation.rule_features),
        [observation.rule_features.length],
      ),
      active_player: int64Tensor(this.Tensor, observation.active_players),
      player_count: int64Tensor(this.Tensor, observation.player_counts),
      round_number: int64Tensor(this.Tensor, observation.rounds),
      action_sources: int64Tensor(this.Tensor, actions.map((action) => action.source)),
      action_targets: int64Tensor(this.Tensor, actions.map((action) => action.target)),
      action_kinds: int64Tensor(
        this.Tensor,
        actions.map((action) => ACTION_KIND_CODES[action.kind]),
      ),
      action_parameters: int64Tensor(
        this.Tensor,
        actions.map((action) => action.parameter),
      ),
    };
    const started = performance.now();
    const outputs = await this.session.run(inputs);
    const milliseconds = performance.now() - started;
    const logits = outputs.logits.data as Float32Array;
    let actionIndex = 0;
    for (let index = 1; index < logits.length; index += 1) {
      if (logits[index] > logits[actionIndex]) {
        actionIndex = index;
      }
    }
    return { actionIndex, milliseconds, legalActions: logits.length };
  }

  release(): void {
    void this.session.release();
  }
}
