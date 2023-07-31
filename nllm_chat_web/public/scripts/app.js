import {
    createEffect,
    createSignal,
    onCleanup,
} from "https://cdn.skypack.dev/solid-js";
import { render } from "https://cdn.skypack.dev/solid-js/web";
import html from "https://cdn.skypack.dev/solid-js/html";
import { TopBar, Footer } from "./navigation.js";

function App() {
    return html`
        <${TopBar} />
        <${Conversation} />
        <${Footer} />
    `;
}

function Conversation() {
    const [messageInputValue, setMessageInputValue] = createSignal("");
    const [usernameValue, setUsernameValue] = createSignal("user1");
    const [textAreaValue, setTextAreaValue] = createSignal("");
    const [connectDisabled, setConnectDisabled] = createSignal(false);
    const [wsInvokeSend, setWsInvokeSend] = createSignal((_msg) => { });

    const chatTextArea = html`
        <textarea class="block p-2.5 w-full h-[250px] text-sm text-gray-900 bg-gray-50 rounded-lg leading-tight border border-gray-300 focus:ring-green-500 focus:border-green-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-green-500 dark:focus:border-green-500"
            rows="4" readonly>
                ${textAreaValue}
        </textarea>
    `;

    window.onerror = function myErrorHandler(errorMsg, url, lineNumber) {
        setTextAreaValue(value => value + "Error occured: " + errorMsg + "\n\n");
        return false;
    }

    const onConnect = () => {
        if (connectDisabled()) {
            return;
        }

        setConnectDisabled(true);
        const websocket = new WebSocket(document.location.origin.replace(/^http/, "ws") + "/websocket");

        websocket.onopen = function () {
            console.log("connection opened");
            websocket.send(usernameValue());
        }

        websocket.onerror = function (e) {
            setTextAreaValue(value => value + "Connection error occured\n\n");
        }

        websocket.onclose = function () {
            console.log("connection closed");
            setConnectDisabled(false);
        }

        websocket.onmessage = function (e) {
            const chatBotPartialPrefix = "[CHAT_PARTIAL]: ";
            const chatBotCompletedPrefix = "Chat: ";
            const getMatchesOld = value => /(.*\n\n)(Chat: .*?　)(\n\n.*?)/.exec(value);
            const getMatches = value => {
                const padIdx = value.lastIndexOf("　");
                if (padIdx < 0) return null;
                const preChatStartIdx = value.substring(0, padIdx).lastIndexOf("\n\nChat: ");
                if (preChatStartIdx < 0) return null;
                const chatStartIdx = preChatStartIdx + 2;
                const chatEndIdx = padIdx + 1;
                return [value, value.substring(0, chatStartIdx), value.substring(chatStartIdx, chatEndIdx), value.substring(chatEndIdx)]
            };

            if (e.data.startsWith(chatBotPartialPrefix)) {
                let partial = e.data.substring(chatBotPartialPrefix.length);
                setTextAreaValue(value => {
                    const matches = getMatches(value);
                    if (matches) {
                        return `${matches[1]}Chat: ${partial}　${matches[3]}`;
                    } else {
                        return value;
                    }
                });
                return;
            }

            if (e.data.startsWith(chatBotCompletedPrefix)) {
                const value = textAreaValue();
                const matches = getMatches(value);
                if (matches) {
                    console.log("received message: " + e.data);
                    setTextAreaValue(value => `${matches[1]}${e.data}${matches[3]}`);
                    return;
                }
            }

            console.log("received message: " + e.data);
            setTextAreaValue(value => value + e.data + "\n\n");
            if (chatTextArea) {
                chatTextArea.scrollTop = chatTextArea.scrollHeight;
            }
        }

        setWsInvokeSend(_ => ({ send: data => websocket.send(data) }));
    };

    const onUsernameKeyDown = e => {
        if (e.key == "Enter") {
            e.preventDefault();
            onConnect();
        }
    };

    const onMessageKeyDown = e => {
        if (e.key == "Enter") {
            e.preventDefault();
            onMessageSubmit();
        }
    };

    const onMessageSubmit = () => {
        const { send } = wsInvokeSend();
        const msg = messageInputValue();
        if (send && msg) {
            send(msg);
            setMessageInputValue("");
        }
    };

    return html`
        <div class="flex flex-col max-w-screen-xl mx-auto p-4 space-y-2">
            <h2 class="text-2xl font-extrabold dark:text-white">Chat with others and with AI!</h2>
            
            <div>
                <label for="website-admin" class="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white">Username</label>
                <div class="relative">
                    <div class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                    <svg class="w-4 h-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10 0a10 10 0 1 0 10 10A10.011 10.011 0 0 0 10 0Zm0 5a3 3 0 1 1 0 6 3 3 0 0 1 0-6Zm0 13a8.949 8.949 0 0 1-4.951-1.488A3.987 3.987 0 0 1 9 13h2a3.987 3.987 0 0 1 3.951 3.512A8.949 8.949 0 0 1 10 18Z"/>
                    </svg>
                    </div>
                    <form onsubmit=${onConnect}>
                        <input class="block w-full p-4 pl-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-green-500 focus:border-green-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-green-500 dark:focus:border-green-500"
                            required placeholder="Username" type="text" id="website-admin"
                            value=${usernameValue}
                            onInput=${e => setUsernameValue(e.currentTarget.value)}
                            onKeyDown=${onUsernameKeyDown}>
                    </form>
                    <button class="text-white absolute right-2.5 bottom-2.5 bg-green-700 hover:bg-green-800 focus:ring-4 focus:outline-none focus:ring-green-300 font-medium rounded-lg text-sm px-4 py-2 dark:bg-green-600 dark:hover:bg-green-700 dark:focus:ring-green-800"
                        type="submit" disabled=${connectDisabled} onClick=${onConnect}>
                            Join Chat
                    </button>
                </div>
            </div>

            ${chatTextArea}

            <div>
                <label for="chat" class="sr-only">Your message</label>
                <div class="flex items-center px-3 py-2 rounded-lg border border-gray-300 bg-gray-50 dark:bg-gray-700 dark:border-gray-600">
                    <button type="button" class="inline-flex justify-center p-2 text-gray-500 rounded-lg cursor-pointer hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-600">
                        <svg class="w-5 h-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 18">
                            <path fill="currentColor" d="M13 5.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0ZM7.565 7.423 4.5 14h11.518l-2.516-3.71L11 13 7.565 7.423Z"/>
                            <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 1H2a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1Z"/>
                            <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0ZM7.565 7.423 4.5 14h11.518l-2.516-3.71L11 13 7.565 7.423Z"/>
                        </svg>
                        <span class="sr-only">Upload image</span>
                    </button>
                    <button type="button" class="p-2 text-gray-500 rounded-lg cursor-pointer hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-600">
                        <svg class="w-5 h-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
                            <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.408 7.5h.01m-6.876 0h.01M19 10a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM4.6 11a5.5 5.5 0 0 0 10.81 0H4.6Z"/>
                        </svg>
                        <span class="sr-only">Add emoji</span>
                    </button>
                    <textarea class="block mx-4 p-2.5 w-full text-sm text-gray-900 bg-white rounded-lg border border-gray-300 focus:ring-green-500 focus:border-green-500 dark:bg-gray-800 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-green-500 dark:focus:border-green-500"
                        id="chat" rows="1" placeholder="Your message..."
                        value=${messageInputValue}
                        onKeyDown=${onMessageKeyDown}
                        onInput=${e => setMessageInputValue(e.currentTarget.value)}>
                    </textarea>
                    <button class="inline-flex justify-center p-2 text-green-600 rounded-full cursor-pointer hover:bg-green-100 dark:text-green-500 dark:hover:bg-gray-600"
                        onClick=${onMessageSubmit}>
                        <svg class="w-5 h-5 rotate-90" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 20">
                            <path d="m17.914 18.594-8-18a1 1 0 0 0-1.828 0l-8 18a1 1 0 0 0 1.157 1.376L8 18.281V9a1 1 0 0 1 2 0v9.281l6.758 1.689a1 1 0 0 0 1.156-1.376Z"/>
                        </svg>
                        <span class="sr-only">Send message</span>
                    </button>
                </div>
            </div>
        </div>
    `;
}

render(App, document.body);