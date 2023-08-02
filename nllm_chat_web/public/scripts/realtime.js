///@ts-check

import { storageSetUsername, toCallback } from "./utils.js";

/**
 * @param {{
 *     onConnected: () => void;
 *     onDisconnected: () => void;
 *     onMessage: (msg: string) => void;
 *     onPartialMessage: (event: { completed: boolean; exec: (chatHistory: string) => string[] | null; payload: string; }) => void;
 *     username: () => string;
 * }} options
 */
export function createWebSocket({ onConnected, onDisconnected, onMessage, onPartialMessage, username }) {
    onConnected = toCallback(onConnected);
    onDisconnected = toCallback(onDisconnected);
    onMessage = toCallback(onMessage);
    username = toCallback(username);

    const state = { isConnected: false, send: _data => { } };
    const start = async () => {
        if (state.isConnected) {
            return Promise.resolve(false);
        }

        state.isConnected = true;
        onConnected();
        return new Promise((resolve, reject) => {
            const websocket = new WebSocket(document.location.origin.replace(/^http/, "ws") + "/websocket");

            websocket.onopen = function () {
                console.log("connection opened");
                const usernameValue = username();
                websocket.send(usernameValue);
                storageSetUsername(usernameValue);
                resolve(true);
            };

            websocket.onerror = function (e) {
                onMessage("Connection error occured");
                reject(e);
            };

            websocket.onclose = function () {
                console.log("connection closed");
                onMessage("🚪 You have left the room for now. Tap 'Join Chat' to rejoin...");
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

            state.send = data => websocket.send(data);
        });
    };

    return [start, data => state.send(data)];
}