# `name` is the name of the package as used for `pip install package` (use hyphens)
name = "template-experiment"
# `path` is the name of the package for `import package` (use underscores)
path = name.lower().replace("-", "_").replace(" ", "_")
# Your version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
version = "0.1.dev0"
author = "Finlay Miller"
author_email = "finlay@dal.ca"
description = ""  # One-liner
url = ""  # your project homepage
license = "Unlicense"  # See https://choosealicense.com
