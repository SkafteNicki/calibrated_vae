import os

import slack

if "SLACK" in os.environ:
    _client = slack.WebClient(token=os.environ.get("SLACK"))
else:
    _client = None


def post_message(text: str) -> None:
    if _client is None:
        return

    _client.chat_postMessage(channel="D0368K4PVQE", text=text)
