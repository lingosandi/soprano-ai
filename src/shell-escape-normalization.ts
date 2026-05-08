export function decodeEscapedShellText(value: string): string {
    if (!/\\(?:[nrt'"\\]|u[0-9a-fA-F]{4})/.test(value)) {
        return value
    }

    let decoded = ""
    let escaping = false
    let unicodeEscape = ""

    for (const char of value) {
        if (unicodeEscape.length > 0) {
            unicodeEscape += char

            if (unicodeEscape.length === 5) {
                if (/^u[0-9a-fA-F]{4}$/.test(unicodeEscape)) {
                    decoded += String.fromCharCode(Number.parseInt(unicodeEscape.slice(1), 16))
                } else {
                    decoded += `\\${unicodeEscape}`
                }
                unicodeEscape = ""
            }
            continue
        }

        if (!escaping) {
            if (char === "\\") {
                escaping = true
            } else {
                decoded += char
            }
            continue
        }

        escaping = false

        switch (char) {
            case "n":
                decoded += "\n"
                break
            case "r":
                decoded += "\r"
                break
            case "t":
                decoded += "\t"
                break
            case '"':
                decoded += '"'
                break
            case "'":
                decoded += "'"
                break
            case "\\":
                decoded += "\\"
                break
            case "u":
                unicodeEscape = "u"
                break
            default:
                decoded += `\\${char}`
                break
        }
    }

    if (unicodeEscape.length > 0) {
        decoded += `\\${unicodeEscape}`
    }

    if (escaping) {
        decoded += "\\"
    }

    return decoded
}