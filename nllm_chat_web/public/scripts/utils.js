///@ts-check

const storageKeys = {
    lastUsername: "NLLM__LAST_USERNAME"
};

export function generateDefaultUsername() {
    const initUserIndex = `1${Math.floor(Math.random() * 999)}`;
    const initUserId = `${"00000".substring(initUserIndex.length)}${initUserIndex}`;
    const initUsername = "anon_1" + initUserId;
    return initUsername;
}

export function storageGetUsername() {
    return localStorage.getItem(storageKeys.lastUsername);
}

export function storageSetUsername(username) {
    if (!username || typeof username !== "string") {
        return;
    }
    localStorage.setItem(storageKeys.lastUsername, username.trim());
}

export function toCallback(fnOrValue) {
    return typeof fnOrValue === "function" ? fnOrValue : (() => fnOrValue);
}