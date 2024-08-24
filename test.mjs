import ManagerInit from "./main.mjs"


ManagerInit().then((m) => {
    const builder = new m.VectorPathBuilder();
    builder.Load();

    const t0 = Date.now();
    for (let i = 0; i < 1000; i++) {
        builder.Process();
    }
    const t1 = Date.now();
    console.log(t1 - t0);
})