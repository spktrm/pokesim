const zeroAsciiCode = "0".charCodeAt(0);
const d_AsciiCode = "d".charCodeAt(0);

export function actionCharToString(actionChar: string): string {
    const actionIndex = actionChar.charCodeAt(0);
    const actionIndexMinusOffset = actionIndex - zeroAsciiCode;
    // assuming its the string "0-9"

    if (0 <= actionIndex && actionIndex <= 3) {
        return `move ${actionIndexMinusOffset + 1}`;
    } else if (3 < actionIndex && actionIndex <= 9) {
        return `switch ${actionIndexMinusOffset - 3}`;
    } else if (actionIndex === d_AsciiCode) {
        return `default`;
    } else {
        return `default`;
    }
}
