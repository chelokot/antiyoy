import assert from "node:assert/strict";
import test from "node:test";
import {
  actionAtTarget,
  actionsForIntent,
  indexedActions,
  movableSources,
  resolveHexClick,
} from "../app/game-interaction";
import type { CellView, CoreAction } from "../app/game-types";

const cells = Array.from({ length: 8 }, (_, id): CellView => ({
  id,
  playable: true,
  owner: id < 4 ? 0 : null,
  object: "Empty",
  strength: id === 0 || id === 1 ? 1 : 0,
  ready: id === 0 || id === 1,
  province: id < 4 ? 3 : null,
  defense: 0,
}));

const legalActions: CoreAction[] = [
  "EndTurn",
  { Move: { source: 0, target: 4 } },
  { Move: { source: 1, target: 5 } },
  { Recruit: { province: 2, target: 3, strength: 1 } },
  { Build: { target: 2, structure: "Farm" } },
];

test("a selected unit exposes only destinations belonging to that source", () => {
  const indexed = indexedActions(legalActions);
  const targets = actionsForIntent(indexed, { kind: "move", source: 0 }, cells);
  assert.deepEqual(targets.map(({ action }) => action), [{ Move: { source: 0, target: 4 } }]);
  assert.equal(actionAtTarget(targets, 5), null);
  assert.deepEqual(actionAtTarget(targets, 4)?.action, { Move: { source: 0, target: 4 } });
});

test("shop intents preserve their province and item identity", () => {
  const indexed = indexedActions(legalActions);
  assert.deepEqual(
    actionsForIntent(indexed, { kind: "recruit", provinceCapital: 2, strength: 1 }, cells)
      .map(({ action }) => action),
    [{ Recruit: { province: 2, target: 3, strength: 1 } }],
  );
  assert.deepEqual(
    actionsForIntent(indexed, { kind: "build", province: 3, structure: "Farm" }, cells)
      .map(({ action }) => action),
    [{ Build: { target: 2, structure: "Farm" } }],
  );
});

test("movable sources are derived independently from destination cells", () => {
  assert.deepEqual([...movableSources(indexedActions(legalActions))], [0, 1]);
});

test("a shared destination never changes the selected move source", () => {
  const indexed = indexedActions([
    "EndTurn",
    { Move: { source: 0, target: 4 } },
    { Move: { source: 1, target: 4 } },
  ]);
  const sourceZeroActions = actionsForIntent(indexed, { kind: "move", source: 0 }, cells);
  assert.deepEqual(resolveHexClick(sourceZeroActions, movableSources(indexed), 4), {
    kind: "play",
    actionIndex: 1,
  });
});

test("clicking outside the selected move zone cancels the intent", () => {
  const indexed = indexedActions(legalActions);
  const sourceZeroActions = actionsForIntent(indexed, { kind: "move", source: 0 }, cells);
  assert.deepEqual(resolveHexClick(sourceZeroActions, movableSources(indexed), 7), {
    kind: "cancel",
  });
});
