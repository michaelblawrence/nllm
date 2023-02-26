
a = document.querySelector("body > pre").innerText

JSON.stringify([
    ...new Set(a.split('\n')
        .map(a => /\d{2}\/\d{2}\/\d{4}, \d{2}:\d{2} - .*?: (.*)/.exec(a))
        .filter(x => x)
        .map(a => a[1])
        .flatMap(x => x.toLowerCase()
            .replace(/[',.]/g, "")
            .split(' ')
            .filter(x => /^[a-z]+$/.test(x))
        )
    )
])
