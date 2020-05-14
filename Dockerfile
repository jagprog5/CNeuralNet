# generates 139MB docker image

FROM alpine:3.7
RUN apk add --no-cache musl-dev make ncurses-dev
RUN apk add tcc --no-cache --repository http://dl-cdn.alpinelinux.org/alpine/edge/testing/ --allow-untrusted
WORKDIR /app
COPY . .
# TODO revise makefile for tcc
CMD tcc src/FFNN.c src/FFNNInspection.c src/interfaceUtil.c src/main.c src/MNISTRead.c -Iheaders -lm -lncurses -o cneuralnet && ./cneuralnet
