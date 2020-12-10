from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    """
    Remove HTML tags from text.

    Based this answer on stackoverflow: 
        https://stackoverflow.com/a/925630/8865008.
    """

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()

def strip_tags(text):
    """[summary]

    Args:
        text (string): Text to be stripped.

    Returns:
        string: stripped text.
    """

    html_stripper = MLStripper()
    html_stripper.feed(text)

    return html_stripper.get_data()