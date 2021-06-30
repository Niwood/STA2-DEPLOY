from tabulate import tabulate
from colorama import Fore, Back, Style
from colorama import init as colorama_init
colorama_init()

a = Fore.RED + '123' + Style.RESET_ALL

table = [
    ["Sun", 696000, 1989100000],
    ["Earth", 6371, 5973.6],
    ["Moon", a, 73.5],
    ["Mars", 3390, 641.85]]
headers = ["Planet", "R (km)", "mass (x 10^29 kg)"]


print(tabulate(table, headers, tablefmt="fancy_grid"))