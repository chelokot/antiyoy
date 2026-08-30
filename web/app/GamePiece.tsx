import type { CSSProperties } from "react";
import type { CellView } from "./game-types";

type PieceName =
  | "capital"
  | "farm"
  | "grave"
  | "palm"
  | "pine"
  | "strong-tower"
  | "tower"
  | `unit-${1 | 2 | 3 | 4}`;

function pieceStyle(piece: PieceName): CSSProperties {
  return { backgroundImage: `url("/game-pieces/${piece}.png")` };
}

function Piece({ name, className = "" }: { name: PieceName; className?: string }) {
  return <span className={`game-piece ${className}`} style={pieceStyle(name)} aria-hidden="true" />;
}

function UnitPiece({ strength }: { strength: number }) {
  const normalizedStrength = Math.min(4, Math.max(1, strength)) as 1 | 2 | 3 | 4;
  return <Piece name={`unit-${normalizedStrength}`} className={`game-unit game-unit-${normalizedStrength}`} />;
}

export function GamePiece({ cell }: { cell: CellView }) {
  if (cell.strength > 0) {
    return <UnitPiece strength={cell.strength} />;
  }
  const name = {
    Capital: "capital",
    Farm: "farm",
    Tower: "tower",
    StrongTower: "strong-tower",
    Pine: "pine",
    Palm: "palm",
    Grave: "grave",
  }[cell.object] as PieceName | undefined;
  return name === undefined ? null : <Piece name={name} className={`game-object game-object-${name}`} />;
}

export function ShopPiece({ kind, strength = 0 }: { kind: "unit" | "farm" | "tower" | "strong-tower" | "tree"; strength?: number }) {
  if (kind === "unit") {
    return <UnitPiece strength={strength} />;
  }
  const name: PieceName = kind === "tree" ? "pine" : kind;
  return <Piece name={name} className={`shop-piece shop-piece-${kind}`} />;
}
