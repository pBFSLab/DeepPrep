.PHONY: build
build: butt1d.so

butt1d.so: butt1dfiles/*.f
	cd butt1dfiles && rm -rf butt1d.cpython-*.so && f2py3 -c -m butt1d butt1d.f fftsol.f npad.f
	cd butt1dfiles && rm -rf butt1d.so && mv butt1d.cpython-*.so butt1d.so
	mv butt1dfiles/butt1d.so ./