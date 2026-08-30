import type { CellView, CoreAction } from "./game-types";

export type IndexedAction = {
  action: CoreAction;
  index: number;
};

export type ActionIntent =
  | { kind: "move"; source: number }
  | { kind: "recruit"; provinceCapital: number; strength: number }
  | { kind: "build"; province: number; structure: string }
  | { kind: "plant-tree"; province: number };

export function indexedActions(actions: CoreAction[]): IndexedAction[] {
  return actions.map((action, index) => ({ action, index }));
}

export function actionTarget(action: CoreAction): number | null {
  if (typeof action === "string") {
    return null;
  }
  if ("Move" in action) {
    return action.Move.target;
  }
  if ("Recruit" in action) {
    return action.Recruit.target;
  }
  if ("Build" in action) {
    return action.Build.target;
  }
  if ("PlantTree" in action) {
    return action.PlantTree.target;
  }
  return null;
}

export function actionsForIntent(
  actions: IndexedAction[],
  intent: ActionIntent | null,
  cells: CellView[],
): IndexedAction[] {
  if (intent === null) {
    return [];
  }
  return actions.filter(({ action }) => {
    if (intent.kind === "move") {
      return typeof action !== "string" && "Move" in action && action.Move.source === intent.source;
    }
    if (intent.kind === "recruit") {
      return typeof action !== "string"
        && "Recruit" in action
        && action.Recruit.province === intent.provinceCapital
        && action.Recruit.strength === intent.strength;
    }
    if (intent.kind === "build") {
      return typeof action !== "string"
        && "Build" in action
        && action.Build.structure === intent.structure
        && cells[action.Build.target].province === intent.province;
    }
    return typeof action !== "string"
      && "PlantTree" in action
      && cells[action.PlantTree.target].province === intent.province;
  });
}

export function movableSources(actions: IndexedAction[]): Set<number> {
  return new Set(actions.flatMap(({ action }) =>
    typeof action !== "string" && "Move" in action ? [action.Move.source] : [],
  ));
}

export function actionAtTarget(actions: IndexedAction[], target: number): IndexedAction | null {
  return actions.find(({ action }) => actionTarget(action) === target) ?? null;
}

export function globalActions(actions: IndexedAction[]): IndexedAction[] {
  return actions.filter(({ action }) => actionTarget(action) === null);
}
