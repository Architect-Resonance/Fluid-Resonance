/**
 * linalg_utils.js
 * 
 * Minimal Linear Algebra suite for Spectral Analysis.
 * Optimized for small symmetric matrices (Laplacians).
 */

const LinAlg = {
  // Transpose
  transpose(A) {
    const n = A.length;
    const m = A[0].length;
    let T = Array.from({ length: m }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        T[j][i] = A[i][j];
      }
    }
    return T;
  },

  // Matrix multiplication
  multiply(A, B) {
    const n = A.length;
    const m = B[0].length;
    const p = B.length;
    let C = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < p; k++) {
          sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
      }
    }
    return C;
  },

  // QR decomposition using Householder reflections
  qr(A) {
    const n = A.length;
    let Q = LinAlg.identity(n);
    let R = A.map(row => new Float64Array(row));

    for (let k = 0; k < n - 1; k++) {
      let x = new Float64Array(n - k);
      for (let i = k; i < n; i++) x[i - k] = R[i][k];
      
      let normX = LinAlg.norm(x);
      if (normX < 1e-12) continue;
      
      let v = new Float64Array(n - k);
      v[0] = x[0] + (x[0] >= 0 ? 1 : -1) * normX;
      for (let i = 1; i < n - k; i++) v[i] = x[i];
      
      let normV = LinAlg.norm(v);
      for (let i = 0; i < n - k; i++) v[i] /= normV;

      // Update R: R = (I - 2vv^T)R
      for (let j = k; j < n; j++) {
        let dot = 0;
        for (let i = k; i < n; i++) dot += v[i - k] * R[i][j];
        for (let i = k; i < n; i++) R[i][j] -= 2 * v[i - k] * dot;
      }

      // Update Q: Q = Q(I - 2vv^T)
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let j = k; j < n; j++) dot += Q[i][j] * v[j - k];
        for (let j = k; j < n; j++) Q[i][j] -= 2 * dot * v[j - k];
      }
    }
    return { Q, R };
  },

  identity(n) {
    return Array.from({ length: n }, (_, i) => {
      let row = new Float64Array(n);
      row[i] = 1;
      return row;
    });
  },

  norm(v) {
    let s = 0;
    for (let x of v) s += x * x;
    return Math.sqrt(s);
  },

  // QR algorithm for eigenvalues of a symmetric matrix
  eigvalsh(A, iterations = 60) {
    let Ak = A.map(row => new Float64Array(row));
    for (let i = 0; i < iterations; i++) {
        const { Q, R } = LinAlg.qr(Ak);
        Ak = LinAlg.multiply(R, Q);
    }
    return Ak.map((row, i) => row[i]).sort((a,b) => a-b);
  },

  // Minimal SVD for finding Null Space
  // Based on Eigen-decomposition of A^T * A
  svd(A) {
    const At = LinAlg.transpose(A);
    const AtA = LinAlg.multiply(At, A);
    const V_eig = LinAlg.eigvectors(AtA);
    const singularValues = V_eig.map(e => Math.sqrt(Math.max(0, e.value)));
    
    return { 
        V: V_eig.map(e => e.vector), 
        S: singularValues 
    };
  },

  // Extract Null Space basis
  nullSpace(A, tol = 1e-4) {
    const { V, S } = LinAlg.svd(A);
    return V.filter((_, i) => S[i] < tol);
  },

  // Helper: Eigenvectors + Eigenvalues using QR algorithm with Wilkinson Shifts
  eigvectors(A, iterations = 300) {
    const n = A.length;
    let Ak = A.map(row => new Float64Array(row));
    let Q_total = LinAlg.identity(n);

    for (let k = 0; k < iterations; k++) {
        // Wilkinson Shift for faster convergence
        // Decides a shift 'mu' from the bottom-right 2x2 submatrix
        const m = n - 1;
        const a = Ak[m-1][m-1];
        const b = Ak[m-1][m];
        const c = Ak[m][m];
        const d = (a - c) / 2;
        const sign = d >= 0 ? 1 : -1;
        const mu = c - sign * (b * b) / (Math.abs(d) + Math.sqrt(d * d + b * b));

        // Shift
        for (let i = 0; i < n; i++) Ak[i][i] -= mu;

        const { Q, R } = LinAlg.qr(Ak);
        Ak = LinAlg.multiply(R, Q);

        // Unshift
        for (let i = 0; i < n; i++) Ak[i][i] += mu;
        
        Q_total = LinAlg.multiply(Q_total, Q);
        
        // Check for convergence (off-diagonal elements near zero)
        let offDiag = 0;
        for (let i = 0; i < n-1; i++) offDiag += Math.abs(Ak[i+1][i]);
        if (offDiag < 1e-14) break;
    }
    
    const results = [];
    for (let i = 0; i < n; i++) {
        results.push({
            value: Ak[i][i],
            vector: Q_total.map(row => row[i])
        });
    }
    return results.sort((a, b) => a.value - b.value);
  }

};

module.exports = LinAlg;

