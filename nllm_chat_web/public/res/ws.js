initKeepAliveSocket = () => {
    const keepalive = new WebSocket("ws://localhost:3000/keepalive");
    keepalive.onopen = function () {
        console.log("keepalive connection established.. will restart when code changed");
    };
    keepalive.onclose = function () {
        let restartConnection = {
            restart: () => { },
            handle: null
        }

        restartConnection.restart = () => {
            if (restartConnection.handle) {
                clearTimeout(restartConnection.handle);
                restartConnection.handle = null;
            }
            console.log("connection to server lost, attempting to reestablish connection...");
            restartConnection.handle = setTimeout(() => {
                const keepalive = new WebSocket("ws://localhost:3000/keepalive");
                keepalive.onopen = function () {
                    console.log("server restarted, reloading page...");
                    window.location.reload();
                };
                keepalive.onclose = function () {
                    restartConnection.restart();
                };
            }, 3000);
        };

        restartConnection.restart();
    }
};

initKeepAliveSocket();