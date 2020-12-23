#include <linux/device-mapper.h>
#include <linux/module.h>
#include <linux/time.h>
#include <linux/init.h>
#include <linux/bio.h>
#include <linux/string_helpers.h>
#include <linux/kernel.h>
#include <linux/completion.h>
#include <linux/vmalloc.h>

#define pbas_in_bio          bio_segments

#define bio_begin_lba(bio)   ((bio)->bi_iter.bi_sector)
#define bio_end_lba          bio_end_sector

#define bio_begin_pba(bio)   (lba_to_pba(bio_begin_lba(bio)))
#define bio_end_pba(bio)     (lba_to_pba(bio_end_lba(bio)))

#define MIN_DISK_SIZE (76LL << 10)
#define MAX_DISK_SIZE (10LL << 40)

#define LBA_SIZE 512
#define PBA_SIZE 4096
#define LBAS_IN_PBA (PBA_SIZE / LBA_SIZE)

#define lba_to_pba(lba) ((pba_t) (lba / LBAS_IN_PBA))
#define pba_to_lba(pba) (((lba_t) pba) * LBAS_IN_PBA)

typedef sector_t lba_t;
typedef int32_t pba_t;