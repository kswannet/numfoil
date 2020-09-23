# Copyright 2020 Kilian Swannet, San Kilkis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Entry point for setuptools, configuration is in setup.cfg."""

from pathlib import Path

from setuptools import setup

with open(Path(__file__).parent / ".version") as f:
    __version__ = f.readline()

if __name__ == "__main__":
    setup(
        use_scm_version={
            "fallback_version": __version__,
            "write_to": ".version",
            "write_to_template": "{version}",
        }
    )
