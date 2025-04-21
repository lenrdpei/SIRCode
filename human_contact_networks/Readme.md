# Some human contact networks

## A contact network in a US high school, 2010

* https://doi.org/10.1073/pnas.1009094108
* Reference: Marcel Salathé, Maria Kazandjieva, Jung Woo Lee, Philip Levis, Marcus W. Feldman, and James H. Jones, "A high-resolution human contact network for infectious disease transmission", PNAS December 21, 2010 107 (51) 22020-22025

## SocioPatterns datasets

A gallery of networks available in http://www.sociopatterns.org/
For example:

### Workplace, 2013

* http://www.sociopatterns.org/datasets/contacts-in-a-workplace/
* Reference: Mathieu Génois, Christian L. Vestergaard, Julie Fournet, André Panisson, Isabelle Bonmarin, and Alain Barrat, "Data on face-to-face contacts in an office building suggest a low-cost vaccination strategy based on community linkers", Network Science 3, 326–347 (2015).

### Workplace, 2015

* http://www.sociopatterns.org/datasets/test/
* Reference:  Mathieu Génois and Alain Barrat, "Can co-location be used as a proxy for face-to-face contacts?", EPJ Data Science 7, 11 (2018).

### High school, 2013

* http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/
* Reference: Rossana Mastrandrea, Julie Fournet, and Alain Barrat, "Contact patterns in a high school: A comparison between data collected using wearable sensors, contact diaries and friendship surveys", PLOS ONE 10, e0136497 (2015).

### Primary school, 2014

* http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/
* Reference: Valerio Gemmetto, Alain Barrat, and Ciro Cattuto, "Mitigation of infectious disease at school: targeted class closure vs school closure", BMC Infectious Diseases 14 (2014)

# A London transit network

* https://www.nature.com/articles/ncomms8723
* Reference: Dane Taylor, Florian Klimm, Heather A. Harrington, Miroslav Kramár, Konstantin Mischaikow, Mason A. Porter & Peter J. Mucha, "Topological data analysis of contagion maps for examining spreading processes on networks", Nature Communications volume 6, Article number: 7723 (2015)

# Preprocessing of contact network data

* The static network data here are preprocessed by the python code "create_realistic_contact_net.py"
  * The original data are not shown in this folder.
  * Note: The code is developed in Python 2.7, where some syntax may differ from Python 3.
* Method: The human contact data in the above examples are based on face-to-face contacts recorded through wearable sensors over a certain period. For example, for SocioPatterns data, each data set contains a lists of active contacts between two individuals lasting for 20 seconds and the membership information of each individual (belonging to a class or department). To build the contact network, we first aggregate the contacts between any two individuals $i$ and $j$, and consider the link between node $i$ and node $j$ as active if the cumulative contact duration between them in the recording period is not less than 60 seconds. The information of total contact duration is also retained.
