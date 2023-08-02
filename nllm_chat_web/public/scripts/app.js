// /@ts-check
import {
    createEffect,
    createSignal,
    mergeProps,
    onCleanup,
} from "https://cdn.skypack.dev/solid-js";
import { render } from "https://cdn.skypack.dev/solid-js/web";
import html from "https://cdn.skypack.dev/solid-js/html";

import { TopBar, Footer } from "./navigation.js";
import { InsertEmojiIcon, SendMessageIcon, UploadImageIcon, UserIcon } from "./svg.js";
import { eventHandler, storageGetUsername, generateDefaultUsername } from "./utils.js";
import { createWebSocket } from "./realtime.js";

function App() {
    return html`
        <${TopBar} />
        <${Conversation} />
        <${Footer} />
    `;
}

function Conversation() {
    const initialUsername = storageGetUsername() || generateDefaultUsername();
    const [messageInputValue, setMessageInputValue] = createSignal("");
    const [usernameValue, setUsernameValue] = createSignal(initialUsername);
    const [textAreaValue, setTextAreaValue] = createSignal("");
    const [connectDisabled, setConnectDisabled] = createSignal(false);

    const chatTextArea = html`
        <textarea class="block p-2.5 w-full h-[250px] text-sm text-gray-900 bg-gray-50 rounded-lg leading-tight border border-gray-300 focus:ring-green-500 focus:border-green-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-green-500 dark:focus:border-green-500"
            rows="4" readonly>
                ${textAreaValue}
        </textarea>
    `;

    const [wsStart, wsSend] = createWebSocket({
        onConnected: () => setConnectDisabled(true),
        onDisconnected: () => setConnectDisabled(false),
        onMessage: msg => {
            setTextAreaValue(value => value + msg + "\n\n");
            chatTextArea.scrollTop = chatTextArea.scrollHeight;
        },
        onPartialMessage: ({ completed: _, exec, payload }) =>
            setTextAreaValue(value => {
                const matches = exec(value);
                return matches ? `${matches[1]}${payload}${matches[3]}` : value;
            }),
        username: () => usernameValue(),
    });

    const onUsernameKeyDown = e => {
        if (e.key == "Enter") {
            e.preventDefault();
            onConnectClick();
        }
    };

    const onConnectClick = () => {
        wsStart();
    }

    const onMessageKeyDown = e => {
        if (e.key == "Enter") {
            e.preventDefault();
            onMessageSubmit();
        }
    };

    const onMessageSubmit = async () => {
        if (!connectDisabled()) {
            try {
                await wsStart();
            }
            catch {
                return;
            }
        }

        const msg = messageInputValue();
        if (msg) {
            wsSend(msg);
            setMessageInputValue("");
        }
    };

    window.onerror = (errorMsg, _url, _lineNumber) => {
        setTextAreaValue(value => value + "Error occurred: " + errorMsg + "\n\n");
        return false;
    }

    return html`
        <div class="flex flex-col max-w-screen-xl mx-auto p-4 space-y-2">

            <h2 class="text-2xl font-extrabold dark:text-white">
                Chat with others and with AI!
            </h2>
            
            <div>
                <label for="website-admin" class="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white">Username</label>
                <div class="relative">
                    <div class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                        <${UserIcon} />
                    </div>
                    <form onsubmit=${onConnectClick}>
                        <input class="block w-full p-4 pl-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-green-500 focus:border-green-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-green-500 dark:focus:border-green-500"
                            required placeholder="Username" type="text" id="website-admin"
                            value=${usernameValue}
                            onInput=${e => setUsernameValue(e.currentTarget.value)}
                            onKeyDown=${onUsernameKeyDown}>
                    </form>
                    <${Button} disabled=${connectDisabled} onClick=${eventHandler(onConnectClick)} text="Join Chat" />
                </div>
            </div>

            ${chatTextArea}

            <div>
                <label for="chat" class="sr-only">Your message</label>
                <div class="flex items-center px-3 py-2 rounded-lg border border-gray-300 bg-gray-50 dark:bg-gray-700 dark:border-gray-600">
                    <button type="button" class="inline-flex justify-center p-2 text-gray-500 rounded-lg cursor-pointer hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-600">
                        <${UploadImageIcon} />
                        <span class="sr-only">Upload image</span>
                    </button>
                    <button type="button" class="p-2 text-gray-500 rounded-lg cursor-pointer hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-600">
                        <${InsertEmojiIcon} />
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
                        <${SendMessageIcon} />
                        <span class="sr-only">Send message</span>
                    </button>
                </div>
            </div>

        </div>
    `;
}

function Button(props) {
    const merged = mergeProps({ disabled: () => false, text: "Button", onClick: () => { } }, props);

    const connectClassList = () => {
        const enabledClassList = "bg-green-700 hover:bg-green-800 dark:bg-green-600 dark:hover:bg-green-700 focus:ring-4 focus:ring-green-300 dark:focus:ring-green-800";
        const disabledClassList = "bg-gray-200 dark:bg-gray-600 text-gray-300";

        const disabled = typeof merged.disabled === "function" ? merged.disabled() : merged.disabled;
        return { [disabled ? disabledClassList : enabledClassList]: true };
    };

    return html`
        <button class="text-white absolute right-2.5 bottom-2.5 focus:outline-none font-medium rounded-lg text-sm px-4 py-2"
            classList=${connectClassList}
            type="submit" disabled=${merged.disabled} onClick=${merged.onClick}>
                ${merged.text}
        </button>
    `;
}

render(App, document.body);


