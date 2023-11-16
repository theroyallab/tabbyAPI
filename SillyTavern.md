# Running on SillyTavern

To run TabbyAPI on SillyTavern, a few more configuration changes are required on SillyTavern's side.

Make sure to get your TabbyAPI **api key** from `api_config.yml`. The admin key is not useable on SillyTavern.

Inside SillyTavern's `config.conf`, add the following under `const requestOverrides`:

```json
{
    hosts: ["127.0.0.1:5000"],
    headers: {
        Authorization: "Bearer <Your API key>"
    }
}
```

This configuration assumes you're using the default config.yml. If you have customized it, make sure to replace `127.0.0.1:5000` with whatever you set the IP and port to in your config.

What this does is intercepts all your oobabooga requests and appends this API key on top of them for security purposes.

Now, inside SillyTavern, just connect to your TabbyAPI URL using oobabooga's text completion API option!

The fetching of models isn't accurate since ooba uses the top-most model in tge model directory (I am not sure why this is the case) to decide which model is loaded. Please make sure to **load a model** beforehand!

Completions work without an issue, just know that all samplers in oobabooga's slider panel won't work in TabbyAPI yet.

Hopefully, Cohee and Ross (or myself) will add full support for TabbyAPI in the not too distant future.
