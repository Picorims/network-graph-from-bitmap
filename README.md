# network-graph-from-bitmap
Converts a bitmap image into a non directed weighted network graph

## Setup

**for Unix/macOS commands visit https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment**

- Python 3.11 is recommended as it was developed using this version.
- Setup a virtual environment:
```
py -m venv env
```

- Activate the environment:
```
.\env\Scripts\activate
```

- Install dependencies:
```
py -m pip install -r requirements.txt
```

- Deactivate the environment:
```
deactivate
```

## Usage

`py network_graph_recognition <path_to_image_file>`

## Pixels color code

- white: background
- black: link networks regardless of their color
- gray: pass-through bridge/tunnel for crossing networks
- same color: same network

## Construction

- a network is a group of lines representing the same transportation system
- a json file **with the same name and location as the image file** structured as below allows to associate a multiplying factor to the values of edges. Its existence is mandatory, although the dictionary can be empty.
```json
{
    "networks_speed": {
        "#bd2c2c": 5,
        "#3184e6": 8
    }
}
```

## Examples

See the samples directory for eczema on how to use this program

## License

MIT license, see license file.