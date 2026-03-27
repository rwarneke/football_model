const NU_EPS = 1e-14;

function poissonPmf(lam: number, maxGoals: number): number[] {
  const pmf = Array.from({ length: maxGoals + 1 }, () => 0);
  if (!(lam > 0)) {
    pmf[0] = 1;
    return pmf;
  }
  pmf[0] = Math.exp(-lam);
  for (let k = 1; k <= maxGoals; k += 1) {
    pmf[k] = pmf[k - 1] * (lam / k);
  }
  return pmf;
}

function normalizeScoreMatrix(matrix: number[][]): number[][] {
  let total = 0;
  for (const row of matrix) {
    for (const value of row) {
      if (Number.isFinite(value)) {
        total += value;
      }
    }
  }
  if (!(total > 0)) {
    return matrix.map((row) => row.map(() => 0));
  }
  const inv = 1 / total;
  return matrix.map((row) => row.map((value) => (Number.isFinite(value) ? value * inv : 0)));
}

export function buildScoreMatrix(params: {
  nu: number;
  lamH: number;
  lamA: number;
  maxGoals: number;
}): number[][] {
  const { nu, lamH, lamA, maxGoals } = params;
  const K = Math.max(0, Math.floor(maxGoals));
  const scoreMatrix = Array.from({ length: K + 1 }, () => Array(K + 1).fill(0));

  if (!(lamH > 0) || !(lamA > 0) || !(nu >= 0)) {
    return scoreMatrix;
  }

  const pX = poissonPmf(lamH, K);
  const pY = poissonPmf(lamA, K);

  if (nu < NU_EPS) {
    let tailX = 1;
    let tailY = 1;
    for (let i = 0; i < K; i += 1) {
      tailX -= pX[i] ?? 0;
      tailY -= pY[i] ?? 0;
    }
    for (let i = 0; i < K; i += 1) {
      const px = pX[i] ?? 0;
      for (let j = 0; j < K; j += 1) {
        scoreMatrix[i][j] = px * (pY[j] ?? 0);
      }
      scoreMatrix[i][K] = px * tailY;
    }
    for (let j = 0; j < K; j += 1) {
      scoreMatrix[K][j] = tailX * (pY[j] ?? 0);
    }
    scoreMatrix[K][K] = tailX * tailY;
    return normalizeScoreMatrix(scoreMatrix);
  }

  const pU = poissonPmf(nu, K);

  for (let u = 0; u < K; u += 1) {
    const pu = pU[u] ?? 0;
    if (pu === 0) {
      continue;
    }
    for (let i = u; i < K; i += 1) {
      const px = pX[i - u] ?? 0;
      if (px === 0) {
        continue;
      }
      for (let j = u; j < K; j += 1) {
        scoreMatrix[i][j] += pu * px * (pY[j - u] ?? 0);
      }
    }
  }

  const cdfX = new Array<number>(K + 1);
  const cdfY = new Array<number>(K + 1);
  let runningX = 0;
  let runningY = 0;
  for (let i = 0; i <= K; i += 1) {
    runningX += pX[i] ?? 0;
    runningY += pY[i] ?? 0;
    cdfX[i] = runningX;
    cdfY[i] = runningY;
  }

  const tailX = new Array<number>(K + 1).fill(1);
  const tailY = new Array<number>(K + 1).fill(1);
  for (let i = 1; i <= K; i += 1) {
    tailX[i] = 1 - (cdfX[i - 1] ?? 0);
    tailY[i] = 1 - (cdfY[i - 1] ?? 0);
  }

  for (let j = 0; j < K; j += 1) {
    let prob = 0;
    for (let u = 0; u <= j; u += 1) {
      const pu = pU[u] ?? 0;
      if (pu === 0) {
        continue;
      }
      prob += pu * (pY[j - u] ?? 0) * (tailX[K - u] ?? 0);
    }
    scoreMatrix[K][j] = prob;
  }

  for (let i = 0; i < K; i += 1) {
    let prob = 0;
    for (let u = 0; u <= i; u += 1) {
      const pu = pU[u] ?? 0;
      if (pu === 0) {
        continue;
      }
      prob += pu * (pX[i - u] ?? 0) * (tailY[K - u] ?? 0);
    }
    scoreMatrix[i][K] = prob;
  }

  let mass = 0;
  for (let i = 0; i < K; i += 1) {
    for (let j = 0; j < K; j += 1) {
      mass += scoreMatrix[i][j] ?? 0;
    }
  }
  for (let j = 0; j < K; j += 1) {
    mass += scoreMatrix[K][j] ?? 0;
  }
  for (let i = 0; i < K; i += 1) {
    mass += scoreMatrix[i][K] ?? 0;
  }
  scoreMatrix[K][K] = Math.max(0, 1 - mass);

  return normalizeScoreMatrix(scoreMatrix);
}
