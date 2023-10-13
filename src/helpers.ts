const zeroAsciiCode = "0".charCodeAt(0);
const d_AsciiCode = "d".charCodeAt(0);

export function actionCharToString(actionChar: string): string {
    const actionIndex = actionChar.charCodeAt(0);
    const actionIndexMinusOffset = actionIndex - zeroAsciiCode;
    // assuming its the string "0-9"

    if (0 <= actionIndexMinusOffset && actionIndexMinusOffset <= 3) {
        return `move ${actionIndexMinusOffset + 1}`;
    } else if (3 < actionIndexMinusOffset && actionIndexMinusOffset <= 9) {
        return `switch ${actionIndexMinusOffset - 3}`;
    } else if (actionIndex === d_AsciiCode) {
        return `default`;
    } else {
        return `default`;
    }
}

export class AsyncQueue {
    private queue: string[];
    private resolveWaitingDequeue?: (value: string) => void;

    constructor() {
        this.queue = [];
    }

    enqueue(item: string): void {
        if (this.resolveWaitingDequeue) {
            this.resolveWaitingDequeue(item);
            this.resolveWaitingDequeue = undefined;
        } else {
            this.queue.push(item);
        }
    }

    async dequeue(): Promise<string> {
        if (this.queue.length > 0) {
            return this.queue.shift()!;
        }

        return new Promise<string>((resolve) => {
            this.resolveWaitingDequeue = resolve;
        });
    }
}

export const delay = (t: number | undefined, val: any = 0) =>
    new Promise((resolve) => setTimeout(resolve, t, val));
