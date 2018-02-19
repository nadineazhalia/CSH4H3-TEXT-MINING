file_berita = open("berita.txt", "r")

berita = file_berita.read()
berita = berita.split()
berita = [x.lower() for x in berita]
berita = list(set(berita))
berita = sorted(berita)

print (berita)