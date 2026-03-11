const HodgeSimulator = require('./FLUID_SIMULATOR_HODGE');

function runVerification() {
    console.log("=== PROJECT ENTROPY: HODGE VERIFICATION ===\n");

    const hub = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1], [5, 0, 3], [6, 2, 4], [7, 5, 6]];
    const spoke = hub.map(c => c.map(v => v + 8));
    const bridge = [[0, 8, 1], [1, 9, 2], [4, 12, 10]];
    const allClauses = [...hub, ...spoke, ...bridge];

    const sim = new HodgeSimulator(allClauses, 16);
    const obs = sim.calculateObservables();

    // Benchmarks from path3_phase_b.py
    const expected = {
        b1: 6,
        euler: -5,
        stokesGap: 0.904723
    };

    console.log(`Checking Betti number b1... ${obs.b1 === expected.b1 ? "PASS" : "FAIL"}`);
    console.log(`Checking Euler Characteristic... ${obs.eulerCharacteristic === expected.euler ? "PASS" : "FAIL"}`);
    console.log(`Checking Stokes Gap... Expect ~${expected.stokesGap}, Got ${obs.stokesGap.toFixed(6)}`);

    const gapDiff = Math.abs(obs.stokesGap - expected.stokesGap);
    if (obs.b1 === expected.b1 && obs.eulerCharacteristic === expected.euler && gapDiff < 0.001) {
        console.log("\nVERIFICATION SUCCESS: Simplicial logic aligns with Research Path.");
        process.exit(0);
    } else {
        console.error("\nVERIFICATION FAILED: Results do not match Python benchmarks.");
        process.exit(1);
    }
}

runVerification();
