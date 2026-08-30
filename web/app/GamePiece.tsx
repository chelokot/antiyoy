import {
  Castle,
  Cross,
  House,
  Shield,
  ShieldCheck,
  Sword,
  Swords,
  Trees,
  TreePine,
} from "lucide-react";
import type { CellView } from "./game-types";

function UnitPiece({ strength }: { strength: number }) {
  const Icon = [Sword, Sword, Shield, ShieldCheck, Swords][strength] ?? Swords;
  return <span className={`game-piece game-unit game-unit-${strength}`}><Icon aria-hidden="true" strokeWidth={2.6} /></span>;
}

export function GamePiece({ cell }: { cell: CellView }) {
  if (cell.strength > 0) {
    return <UnitPiece strength={cell.strength} />;
  }
  const Icon = {
    Capital: Castle,
    Farm: House,
    Tower: Shield,
    StrongTower: ShieldCheck,
    Pine: TreePine,
    Palm: Trees,
    Grave: Cross,
  }[cell.object];
  return Icon === undefined
    ? null
    : <span className={`game-piece game-object game-object-${cell.object.toLowerCase()}`}><Icon aria-hidden="true" strokeWidth={2.5} /></span>;
}

export function ShopPiece({ kind, strength = 0 }: { kind: "unit" | "farm" | "tower" | "strong-tower" | "tree"; strength?: number }) {
  if (kind === "unit") {
    return <UnitPiece strength={strength} />;
  }
  const Icon = {
    farm: House,
    tower: Shield,
    "strong-tower": ShieldCheck,
    tree: TreePine,
  }[kind];
  return <span className={`game-piece shop-piece shop-piece-${kind}`}><Icon aria-hidden="true" strokeWidth={2.5} /></span>;
}
