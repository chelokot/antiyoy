export function GET(request: Request) {
  return Response.redirect(new URL("/game-pieces/capital.png", request.url), 308);
}
