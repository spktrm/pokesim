const zeroAsciiCode = "0".charCodeAt(0);
const threeAsciiCode = "3".charCodeAt(0);
const nineAsciiCode = "9".charCodeAt(0);
const r_AsciiCode = "r".charCodeAt(0);
const d_AsciiCode = "d".charCodeAt(0);

export function actionIndexToString(actionIndex: number) {
    const actionIndexMinusOffset = actionIndex - zeroAsciiCode + 1;
    // assuming its the string "0-9"
    if (zeroAsciiCode <= actionIndex && actionIndex <= threeAsciiCode) {
        return `move ${actionIndexMinusOffset}`;
    } else if (threeAsciiCode < actionIndex && actionIndex <= nineAsciiCode) {
        return `switch ${actionIndexMinusOffset - 4}`;
    } else if (actionIndex === d_AsciiCode) {
        return `default`;
    } else {
        return `default`;
    }
}
