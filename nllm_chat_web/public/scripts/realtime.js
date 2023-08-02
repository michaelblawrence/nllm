///@ts-check

import { storageSetUsername, toCallback, runOnboarding } from "./utils.js";

/**
 * @param {{
 *     onConnected: () => void;
 *     onDisconnected: () => void;
 *     onMessage: (msg: string) => void;
 *     onPartialMessage: (event: { completed: boolean; exec: (chatHistory: string) => string[] | null; payload: string; }) => void;
 *     username: () => string;
 * }} options
 * @returns {[() => Promise<boolean>, (payload: string) => void]}
 */
export function createWebSocket({ onConnected, onDisconnected, onMessage, onPartialMessage, username }) {
    onConnected = toCallback(onConnected);
    onDisconnected = toCallback(onDisconnected);
    onMessage = toCallback(onMessage);
    username = toCallback(username);

    const state = { isConnected: false, locked: false, send: _data => { } };
    const start = async () => {
        if (state.isConnected || state.locked) {
            return Promise.resolve(false);
        }

        state.locked = true;
        return new Promise((resolve, reject) => {
            const websocket = new WebSocket(document.location.origin.replace(/^http/, "ws") + "/websocket");

            websocket.onopen = function () {
                console.log("connection opened");
                const usernameValue = username();
                websocket.send(usernameValue);
                storageSetUsername(usernameValue);

                state.locked = false;
                resolve(true);
            };

            websocket.onerror = function (e) {
                onMessage("There is an issue with your connection. Please try again later...");

                state.locked = false;
                reject(e);
            };

            websocket.onclose = function () {
                if (state.isConnected) {
                    console.log("connection closed");
                    onMessage("🚪 You have left the room for now. Tap 'Join Chat' to rejoin...");
                    state.isConnected = false;
                } else {
                    onMessage("🔌 You are not in a room. Tap 'Join Chat' to join...");
                }
                onDisconnected();
            };

            websocket.onmessage = function (e) {
                const chatBotPartialPrefix = "[CHAT_PARTIAL]: ";
                const chatBotCompletedPrefix = "Chat: ";
                const getMatches = value => {
                    const padIdx = value.lastIndexOf("　");
                    if (padIdx < 0)
                        return null;
                    const preChatStartIdx = value.substring(0, padIdx).lastIndexOf("\n\nChat: ");
                    if (preChatStartIdx < 0)
                        return null;
                    const chatStartIdx = preChatStartIdx + 2;
                    const chatEndIdx = padIdx + 1;
                    return [value, value.substring(0, chatStartIdx), value.substring(chatStartIdx, chatEndIdx), value.substring(chatEndIdx)];
                };

                if (e.data.startsWith(chatBotPartialPrefix)) {
                    const partial = e.data.substring(chatBotPartialPrefix.length);
                    const exec = x => getMatches(x);
                    onPartialMessage({ completed: false, exec, payload: `Chat: ${partial}　` });
                    return;
                }

                if (e.data.startsWith(chatBotCompletedPrefix)) {
                    if (!state.isConnected) {
                        state.isConnected = true;
                        onConnected();
                        onMessage("👋 You have joined the room. You can now talk to others users, and to our AI called 'Chat'");
                        runOnboarding("PROMPT_TO_START_CONVO", () => onMessage("💡 To get started, why not introduce yourself or say hello to 'Chat'?"));
                    }
                    let wasMatch = false;
                    const exec = x => {
                        const matches = getMatches(x);
                        wasMatch = wasMatch || !!matches;
                        return matches;
                    };
                    onPartialMessage({ completed: true, exec, payload: e.data });
                    if (wasMatch) {
                        console.log("received message: " + e.data);
                        return;
                    }
                }

                console.log("received message: " + e.data);
                onMessage(e.data);
            };

            state.send = data => {
                try {
                    websocket.send(data);
                    return true;
                } catch (error) {
                    console.error("Failed to send user message", error);
                    onMessage("😢 Message could not send. Please try send again, or refresh the page");
                    return false;
                }
            };
        });
    };

    return [start, data => state.send(data)];
}