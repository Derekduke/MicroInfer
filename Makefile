DIR_INC = ./inc
DIR_PORT_INC = ./port
DIR_SRC = ./src
DIR_SRC_LAYERS = ./src/layers
DIR_OBJ = ./obj
DIR_BIN = ./bin

SRC = $(wildcard ${DIR_SRC}/*.c)
SRC_LAYERS = $(wildcard ./src/layers/*.c) 
OBJ = $(patsubst %.c,${DIR_OBJ}/%.o,$(notdir ${SRC}))
OBJ_LAYERS = $(patsubst %.c,${DIR_OBJ}/%.o,$(notdir ${SRC_LAYERS}))

BIN_TARGET = ${DIR_BIN}

CC = gcc
CFLAGS = -g -Wall -I${DIR_INC} -I${DIR_PORT_INC}

${BIN_TARGET}:${OBJ} ${OBJ_LAYERS}
	$(CC) $^ $(LIB) -o $@

${DIR_OBJ}/%.o:${DIR_SRC}/%.c
	$(CC) $(CFLAGS) -c  $< -o $@

${DIR_OBJ}/%.o:${DIR_SRC_LAYERS}/%.c
	$(CC) $(CFLAGS) -c  $< -o $@

.PHONY:clean
clean:
#	find ${DIR_OBJ} -name *.o -exec del -rf {}
	del obj\*.o
	del bin.exe