from urllib.request import urlopen

from bs4 import BeautifulSoup


# location entweder vom user erfragen oder über Handyortung
# alternativ aponet
# alternativ gelbe Seiten
def telefonbuch(location):
    location = location.replace(" ", "%20")
    url = 'http://www.dastelefonbuch.de/Apotheken-Notdienste/%s' % location
    soup = BeautifulSoup(urlopen(url), "lxml")
    names = [el["title"] for el in soup.findAll("div", attrs={'class': "name"})]
    adds = [el["title"].replace('Postfach null', '') for el in soup.findAll("a", attrs={'class': "addr"})]
    phone = [el.get_text().strip('\t\n\r') for el in soup.findAll('span', attrs='fon nr')]
    distance = [el.get_text() for el in soup.findAll("div", attrs={'class': "distance"})]
    open = [el.get_text().strip('\t\n\r') for el in soup.findAll("a", attrs={'class': "times tb3_det_link"})]
    return names, adds, phone, distance, open


def aponet(location):
    location = location.lower()
    location = location.replace(" ", "%2B")
    url = 'https://www.aponet.de/service/notdienstapotheke-finden/suchergebnis/0/%s.html' % location

    soup = BeautifulSoup(urlopen(url), "lxml")
    print(soup)
    adds = [el.get_text().strip('\t\n\r') for el in soup.findAll("p", attrs={'class': "adress"})]
    return adds


print(aponet("anzing"))
# names, adds, phone, distance, open = (telefonbuch("anzing"))
# for i in zip(names, adds, phone, distance, open):
#     print(i)
