SUBDIRS := Core Math Features ActionRecognition

all: build

.PHONY: build
build: $(SUBDIRS)

.PHONY: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) --directory=$@

.PHONY: clean
clean:
	@for d in $(SUBDIRS); do cd $$d; $(MAKE) clean; cd ..; done
