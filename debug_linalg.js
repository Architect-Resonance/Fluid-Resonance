const LinAlg = require('./linalg_utils');

const A = [
    [1, -1, 0],
    [0, 1, -1],
    [-1, 0, 1]
]; // Rank 2, nullspace 1 (all 1s)

console.log("Testing Nullspace of 3x3 cyclic graph...");
const ns = LinAlg.nullSpace(A);
console.log("Nullspace Dim (Expected 1):", ns.length);
if (ns.length > 0) {
    console.log("Basis Vector:", ns[0]);
}

// 40x40 Test (Identity)
const I = LinAlg.identity(40);
const nsI = LinAlg.nullSpace(I);
console.log("Nullspace of 40x40 Identity (Expected 0):", nsI.length);
