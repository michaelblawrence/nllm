import {
    createEffect,
    createSignal,
    onCleanup,
} from "https://cdn.skypack.dev/solid-js";
import { render } from "https://cdn.skypack.dev/solid-js/web";
import html from "https://cdn.skypack.dev/solid-js/html";

function App() {
    const [messageInputValue, setMessageInputValue] = createSignal("");
    const [usernameValue, setUsernameValue] = createSignal("user1");
    const [textAreaValue, setTextAreaValue] = createSignal("");
    const [connectDisabled, setConnectDisabled] = createSignal(false);
    const [wsInvokeSend, setWsInvokeSend] = createSignal((_msg) => { });

    const chatTextArea = html`<textarea style="display:block; width:600px; height:400px; box-sizing: border-box" cols="30" rows="10">
        ${textAreaValue}
    </textarea>`;

    if (initKeepAliveSocket) {
        initKeepAliveSocket();
    }

    const onConnect = () => {
        setConnectDisabled(true);
        const websocket = new WebSocket("ws://localhost:3000/websocket");

        websocket.onopen = function () {
            console.log("connection opened");
            websocket.send(usernameValue());
        }

        websocket.onclose = function () {
            console.log("connection closed");
            setConnectDisabled(false);
        }

        websocket.onmessage = function (e) {
            const chatBotPartialPrefix = "[CHAT_PARTIAL]: ";
            const chatBotCompletedPrefix = "Chat: ";
            const getMatchesOld = value => /(.*\r\n)(Chat: .*?　)(\r\n.*?)/.exec(value);
            const getMatches = value => {
                const padIdx = value.lastIndexOf("　");
                if (padIdx < 0) return null;
                const preChatStartIdx = value.substring(0, padIdx).lastIndexOf("\r\nChat: ");
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
            setTextAreaValue(value => value + e.data + "\r\n");
            if (chatTextArea) {
                chatTextArea.scrollTop = chatTextArea.scrollHeight;
            }
        }

        setWsInvokeSend(_ => ({ send: data => websocket.send(data) }));
    };

    const onMessageKeyDown = e => {
        setMessageInputValue(e.currentTarget.value);
        if (e.key == "Enter") {
            const { send } = wsInvokeSend();
            const msg = messageInputValue();
            send(msg);
            setMessageInputValue("")
        }
    };

    return html`
        <h1>ChatNLLM - chat with others and with AI!</h1>

        <input style="display:block; width:100px; box-sizing: border-box" type="text" placeholder="username" 
            value=${usernameValue}
            onInput=${(e) => setUsernameValue(e.currentTarget.value)}>
        <button type="button" disabled=${connectDisabled} onClick=${onConnect}>
            Join Chat
        </button>
        ${chatTextArea}
        <input style="display:block; width:600px; box-sizing: border-box" type="text" placeholder="chat" 
            value=${messageInputValue}
            onKeyDown=${onMessageKeyDown}>
    `;
}
render(App, document.body);