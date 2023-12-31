///@ts-check

const storageKeys = {
    lastUsername: "NLLM__LAST_USERNAME",
    onboardingProgress: "NLLM__ONBOARDING_PROGRESS"
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

/**
 * @param {string} key
 * @param {Function} fn
 */
export function runOnboarding(key, fn) {
    const onboardingTask = getOnboardingTask(key);

    if (!onboardingTask.completed) { 
        fn();
        onboardingTask.complete();
    }
}

/**
 * @param {string} key
 */
export function getOnboardingTask(key) {
    let storage_key = key.toUpperCase();
    const state = storageGetOnboardingProgress();

    return {
        storage_key, 
        completed: !!state[storage_key],
        complete: () => {
            if (!state[storage_key]) {
                state[storage_key] = true;
                storageSetOnboardingProgress(state);
            }
        }
    };
}

/**
 * @param {Function} fn
 */
export function eventHandler(fn) {
    return (() => fn);
}

function storageGetOnboardingProgress() {
    try {
        const json = localStorage.getItem(storageKeys.onboardingProgress) || "{}";
        return JSON.parse(json);
    }
    catch {
        return {};
    }
}
function storageSetOnboardingProgress(state) {
    const json = JSON.stringify(state || {});
    localStorage.setItem(storageKeys.onboardingProgress, json);
}