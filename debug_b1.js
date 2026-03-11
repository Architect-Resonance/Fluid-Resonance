const LinAlg = require('./linalg_utils');

function buildB1() {
    const hub = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1], [5, 0, 3], [6, 2, 4], [7, 5, 6]];
    const spoke = hub.map(c => c.map(v => v + 8));
    const bridge = [[0, 8, 1], [1, 9, 2], [4, 12, 10]];
    const allClauses = [...hub, ...spoke, ...bridge];

    const edgesSet = new Set();
    allClauses.forEach(clause => {
        for (let i = 0; i < clause.length; i++) {
            for (let j = i + 1; j < clause.length; j++) {
                const e = [Math.min(clause[i], clause[j]), Math.max(clause[i], clause[j])];
                edgesSet.add(e.join(','));
            }
        }
    });
    const edges = Array.from(edgesSet).map(s => s.split(',').map(Number)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    const nEdges = edges.length;
    const nVars = 16;
    const B1 = Array.from({ length: nVars }, () => new Float64Array(nEdges));
    edges.forEach(([u, v], i) => {
        B1[u][i] = -1;
        B1[v][i] = 1;
    });
    return B1;
}

const B1 = buildB1();
console.log(`B1 shape: ${B1.length}x${B1[0].length}`);

const start = Date.now();
const { V, S } = LinAlg.svd(B1);
const duration = Date.now() - start;

console.log(`SVD duration: ${duration}ms`);
console.log("Singular Values (top 20):", S.slice(-20)); // Smallest go last in my sort
console.log("Singular Values (bottom 30):", S.slice(0, 30)); // Smallest go first

const nullDim = S.filter(s => s < 1e-4).length;
console.log(`Nullspace Dim (tol 1e-4): ${nullDim}`);

const nullDimStrict = S.filter(s => s < 1e-8).length;
console.log(`Nullspace Dim (tol 1e-8): ${nullDimStrict}`);
