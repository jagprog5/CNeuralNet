# generates 138MB docker image containing all reasources to compile the project

FROM alpine:3.11
RUN apk add --no-cache musl-dev ncurses-dev
RUN apk add tcc --no-cache --repository http://dl-cdn.alpinelinux.org/alpine/edge/testing/ --allow-untrusted
WORKDIR /app
COPY . .
# TODO revise makefile for tcc
CMD tcc src/FFNN.c src/FFNNInspection.c src/interfaceUtil.c src/main.c src/MNISTRead.c -DDOCKER -Iheaders -lm -lncurses -o cneuralnet && ./cneuralnet
