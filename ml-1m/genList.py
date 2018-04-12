outfile = open('kwList.txt', 'w')

with open('m_movies.dat', 'r') as infile:
    for l in infile:
        l.rstrip('\n')
        llist = l.split('::')
        outfile.write(':' + ''.join([c for c in llist[1] if c != ':']) + ' trailer\n')
