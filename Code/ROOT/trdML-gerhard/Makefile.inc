
all: gridfiles.md5

gridfiles.xml: query.sh
	./$< > $@

gridfiles.md5: gridfiles.xml
	xsltproc /alice/data/util/xml2md5.xsl $< > $@

download: $(shell cut -c 49- gridfiles.md5)

/alice/cern.ch/%:
	mkdir -p $(dir $@)
	alien_cp alien:$@ file:$@
