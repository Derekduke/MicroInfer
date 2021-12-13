/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_PORT_H_
#define __MICROINFER_PORT_H_

//#include <stdlib.h>
//#include <stdio.h>

//#define microinfer_malloc(n)    malloc(n)
//#define microinfer_free(n)      free(n)

#define microinfer_memset(a,b,c)     memset(a,b,c)
#define microinfer_memcpy(dst,src,len)  memcpy(dst,src,len)

#define MICROINFER_LOG(...)       printf(__VA_ARGS__)

#endif