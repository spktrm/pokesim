import { actionCharToString } from "./helpers";

export function weightedRandomSample(
    options: any[],
    weights: number[],
    size: number,
): number[] {
    if (options.length !== weights.length) {
        throw new Error(
            "The options and weights arrays must be the same length.",
        );
    }

    const totalWeight = weights.reduce((total, weight) => total + weight, 0);
    const res = [];

    for (let i = 0; i < size; i++) {
        let random = Math.random() * totalWeight;
        let index = 0;

        for (; index < weights.length; index++) {
            random -= weights[index];
            if (random < 0) break;
        }

        res.push(options[index]);
    }

    return res;
}

export function arange(start: number, stop: number, step?: number): number[] {
    const stepSize = step ?? 1;

    if (step === 0) {
        throw new Error("Step should not be zero");
    }

    const output = [];
    if (stepSize > 0) {
        for (let i = start; i < stop; i += stepSize) {
            output.push(i);
        }
    } else {
        for (let i = start; i > stop; i += stepSize) {
            output.push(i);
        }
    }

    return output;
}

export const numArange = arange(0, 10);

export function getRandomAction(legalMask: Int8Array): string {
    const [randIndex] = weightedRandomSample(
        numArange,
        new Array(...legalMask),
        1,
    );
    return actionCharToString(`${randIndex}`);
}

function test() {
    const weights = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0];
    const counts = new Array(weights.length).fill(0);
    const total = 100000;

    const out = weightedRandomSample(numArange, new Array(...weights), total);
    // for (let i = 0; i++; i < total) {
    //     counts[out] += 1;
    // }
    out.map((value) => {
        counts[value] += 1;
    });

    console.log(counts.map((x) => x / total));
}

// test();
