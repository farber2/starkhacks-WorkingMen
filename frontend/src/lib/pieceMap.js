const fallbackGlyphs = {
  K: '♔',
  Q: '♕',
  R: '♖',
  B: '♗',
  N: '♘',
  P: '♙',
  k: '♚',
  q: '♛',
  r: '♜',
  b: '♝',
  n: '♞',
  p: '♟'
};

const imageMap = {
  P: '/assets/pieces/wp.png',
  N: '/assets/pieces/wn.png',
  B: '/assets/pieces/wb.png',
  R: '/assets/pieces/wr.png',
  Q: '/assets/pieces/wq.png',
  K: '/assets/pieces/wk.png',
  p: '/assets/pieces/bp.png',
  n: '/assets/pieces/bn.png',
  b: '/assets/pieces/bb.png',
  r: '/assets/pieces/br.png',
  q: '/assets/pieces/bq.png',
  k: '/assets/pieces/bk.png'
};

export function pieceToAsset(symbol) {
  if (!symbol) {
    return null;
  }
  return imageMap[symbol] ?? null;
}

export function pieceFallback(symbol) {
  if (!symbol) {
    return '';
  }
  return fallbackGlyphs[symbol] ?? symbol;
}
