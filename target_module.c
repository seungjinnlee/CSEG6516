#include "target_module.h"

#define MIN_IOS 16
#define MIN_POOL_PAGES 32

#define RL_GC 1
#define ACTIONS 2
#define episode 1
#define target_state -1
#define gamma 8
#define alpha 1
#define short_max 32768
#define GC_threshold_percentage 99
#define GC_min_interval 100

// define the max number of options of each state
#define previous_interval_choices_max 2
#define current_interval_choices_max 5
#define previous_action_max 5
#define bands_left_max 4
#define pc_utilization_max 5
#define io_type 2
#define state_max previous_interval_choices_max*current_interval_choices_max*previous_action_max*bands_left_max*pc_utilization_max*io_type

// define the action space
// static int32_t action_list[5] = {0, 5, 10, 15, 20};
static int32_t action_list[5] = {0, 2, 4, 6, 8};
static int32_t prev_io_type;
#define action_max 5

// define the state space
struct state{
    int32_t previous_interval;	
    int32_t current_interval;  	
    int32_t previous_action;
    int32_t bands_left;
    int32_t pc_utilization;
    int32_t previous_io_type;
};

// initialize the current state
static struct state current_state={0,0,0,0,0,0};
static int32_t current_action;
static int32_t reward;
static struct state next_state;
static int32_t next_state_max_q;
static int32_t last_io_begin_time=0;
static int32_t io_begin_time=0;
static int32_t io_end_time=0;

//define the Q-Table
static int32_t q[state_max][action_max] = { 0 };
static int32_t explore_count = 0;
static int32_t epsilon1 = 90;
static int32_t epsilon2 = 99;
static int32_t min_pba_in_cache_band = 256;
static int32_t init_complete = 0;


static struct kmem_cache *_io_pool;

struct io {
        struct super_info *super_ctx;
        struct bio *bio;
        struct work_struct work;
        atomic_t pending;
};

struct cache_band {
        int32_t nr;
        pba_t begin_pba;
        pba_t current_pba;

        unsigned long *map;
};

struct super_info {
        struct dm_dev *dev;
        int64_t disk_size;

        int32_t cache_percent;

        int64_t cache_size;
        int32_t nr_cache_bands;
        int64_t usable_size;
        int64_t wasted_size;
        int32_t track_size;
        int64_t band_size;
        int32_t band_size_tracks;
        int32_t band_size_pbas;
        int32_t nr_valid_pbas;
        int32_t nr_usable_pbas;

        pba_t *pba_map;

        int32_t *cache_bands_map;
        struct cache_band *cache_bands;

        int32_t partialGC_band_index;
        int32_t partialGC;
        int32_t* partialGC_band_list;
        int32_t current_partialGC_band_list_index;
        int32_t current_partialGC_band_list_max;

        int32_t nr_bands;
        int32_t nr_usable_bands;
        int32_t cache_assoc;

        mempool_t *io_pool;
        mempool_t *page_pool;
        struct workqueue_struct *queue;
        struct mutex lock;
        struct completion io_completion;
        struct bio_set *bs;
        atomic_t error;
        struct bio **tmp_bios;
        struct bio **rmw_bios;
};

// get the corresponding index of a certain state
static int32_t get_row_q(struct state* s){
    int32_t idx = s->previous_interval*(current_interval_choices_max*previous_action_max*bands_left_max*pc_utilization_max)+\
                s->current_interval*(previous_action_max*bands_left_max*pc_utilization_max)+\
                s->previous_action*(bands_left_max*pc_utilization_max)+\
                s->bands_left*pc_utilization_max+\
                s->pc_utilization*2+\
                s->previous_io_type;
    return idx;
}

// check if the Q-values are undefined for a certain state
static int32_t is_empty_q(struct state* s){
    int32_t flag = 1;
    int32_t row = get_row_q(s);
        int32_t i = 0;
    for(;i<action_max;i++){
        if(q[row][i]!=0)
            flag = 0;
    }
    return flag;
}

// get the max Q-value
static int32_t max_value_q(struct state* s){
    int32_t i=1;
    int32_t row = get_row_q(s);
    int32_t max = q[row][0];
    for(;i<action_max;i++){
        if(q[row][i]>max)
            max = q[row][i];
    }
    return max;
}

// get the max action
static int32_t max_action(struct state* s){
    int32_t i=1,a=0;
    int32_t row = get_row_q(s);
    float max = q[row][0];
    for(;i<action_max;i++){
        if(q[row][i]>max)
            max = q[row][i];
            a = i;
    }
    return a;
}

static int32_t get_reward(int32_t response_time){
    int32_t r = 0;
    // if(response_time<200){
    if(response_time<500){
//     if(response_time<800){
//     if(response_time<1100){
//     if(response_time<1400){
            r = 70;
    }else if(response_time<1300){
            r = 50;
    }else if(response_time<2800){
            r = 0;
    }else{
            r = -100;
    }
    return r;
}

static int32_t current_interval_bins[current_interval_choices_max]=\
        {300, 1000, 3000, 10000, 30000};
        // {300, 1000, 3000, 10000, 30000};

static int32_t calc_current_interval(int32_t c){
    int32_t i = 0;
    for(;i<current_interval_choices_max;i++){
            if(c<current_interval_bins[i]){
                    break;
            }
    }
    return i;
}

static int get_bands_left(int b){
        int r = b/20;
        if(r<3){
                return r;
        }else{
                return 3;
        }
}

static int get_pc_utilization(int min){
    if(min>min_pba_in_cache_band){
        return 4;
    }
    return (min*pc_utilization_max)/min_pba_in_cache_band;

}

static struct state get_next_state(struct state* s, int32_t _current_action, int32_t b, int min){
    struct state ns;
    ns.current_interval = calc_current_interval(io_begin_time-last_io_begin_time);
    if(s->current_interval < 2){
            ns.previous_interval = 0;
    }else{
            ns.previous_interval = 1;
    }
    ns.previous_action = _current_action;
    ns.bands_left = get_bands_left(b);
    ns.pc_utilization = get_pc_utilization(min);
    ns.previous_io_type = prev_io_type;
    return ns;
}

static inline void debug_bio(struct super_info *super_ctx, struct bio *bio, const char *f)

{
        int32_t i;
        unsigned long flags;
        struct bio_vec bv;

        pr_debug("%10s: %c offset: %d size: %u\n",
                 f,
                 (bio_data_dir(bio) == READ ? 'R' : 'W'),
                 bio_begin_pba(bio),
                 bio->bi_iter.bi_size);

        bio_for_each_segment(bv, bio, (bio)->bi_iter) {
                char *addr = bvec_kmap_irq(&bv, &flags);
                pr_debug("seg: %d, addr: %p, len: %u, offset: %u, char: [%d]\n",
                         i, addr, bv.bv_len, bv.bv_offset, *addr);
                bvec_kunmap_irq(addr, &flags);
        }
}

static bool unaligned_bio(struct bio *bio)
{
        return bio_begin_lba(bio) & 0x7 || bio->bi_iter.bi_size & 0xfff;
}

static struct io *alloc_io(struct super_info *super_ctx, struct bio *bio)
{
        struct io *io = mempool_alloc(super_ctx->io_pool, GFP_NOIO);

        if (unlikely(!io)) {
                return NULL;
        }

        memset(io, 0, sizeof(*io));

        io->super_ctx = super_ctx;
        io->bio = bio;

        atomic_set(&io->pending, 0);

        return io;
}

static void dev_d(struct work_struct *work);

static void queue_io(struct io *io)
{
        struct super_info *super_ctx = io->super_ctx;
        struct timeval ct;

        INIT_WORK(&io->work, dev_d);
        last_io_begin_time = io_begin_time;
        do_gettimeofday(&ct);
        io_begin_time = ct.tv_sec*1000+ct.tv_usec/1000;
        queue_work(super_ctx->queue, &io->work);
}

static void release_io(struct io *io, int32_t error)
{
        struct super_info *super_ctx = io->super_ctx;
        bool is_rmw = io->bio == NULL;

        mempool_free(io, super_ctx->io_pool);

        if (is_rmw)
                atomic_set(&super_ctx->error, error);
        else
                bio_endio(io->bio);
}

static inline bool usable_pba(struct super_info *super_ctx, pba_t pba)
{
        return 0 <= pba && pba < super_ctx->nr_usable_pbas;
}

static inline bool usable_band(struct super_info *super_ctx, int32_t band)
{
        return 0 <= band && band < super_ctx->nr_usable_bands;
}

static inline pba_t band_begin_pba(struct super_info *super_ctx, int32_t band)
{
        WARN_ON(!usable_band(super_ctx, band));

        return band * super_ctx->band_size_pbas;
}

static inline pba_t band_end_pba(struct super_info *super_ctx, int32_t band)
{
        return band_begin_pba(super_ctx, band) + super_ctx->band_size_pbas;
}

static inline int32_t pba_band(struct super_info *super_ctx, pba_t pba)
{
        WARN_ON(!usable_pba(super_ctx, pba));

        return pba / super_ctx->band_size_pbas;
}

static inline int32_t bio_band(struct super_info *super_ctx, struct bio *bio)
{
        return pba_band(super_ctx, bio_begin_pba(bio));
}

static inline int32_t band_to_bit(struct super_info *super_ctx, struct cache_band *cb,
                              int32_t band)
{
        return (band - cb->nr) / (super_ctx->nr_cache_bands-1);
}

static inline int32_t bit_to_band(struct super_info *super_ctx, struct cache_band *cb,
                              int32_t bit)
{
        return bit * (super_ctx->nr_cache_bands-1) + cb->nr;
}

static inline struct cache_band *cache_band(struct super_info *super_ctx, int32_t band)
{
        WARN_ON(!usable_band(super_ctx, band));
        return &super_ctx->cache_bands[super_ctx->cache_bands_map[band % (super_ctx->nr_cache_bands-1)]];
}

static inline int32_t free_pbas_in_cache_band(struct super_info *super_ctx,
                                              struct cache_band *cb)
{
        return super_ctx->band_size_pbas - (cb->current_pba - cb->begin_pba);
}

static int32_t pbas_in_band(struct super_info *super_ctx, struct bio *bio, int32_t band)
{
        pba_t begin_pba = max(band_begin_pba(super_ctx, band), bio_begin_pba(bio));
        pba_t end_pba = min(band_end_pba(super_ctx, band), bio_end_pba(bio));

        return max(end_pba - begin_pba, 0);
}

static void unmap_pba_range(struct super_info *super_ctx, pba_t begin, pba_t end)
{
        int32_t i;

        WARN_ON(begin >= end);
        WARN_ON(!usable_pba(super_ctx, end - 1));

        for (i = begin; i < end; ++i)
                super_ctx->pba_map[i] = -1;
}

static pba_t map_pba_range(struct super_info *super_ctx, pba_t begin, pba_t end)
{
        pba_t i;
        int32_t b;
        struct cache_band *cb;

        WARN_ON(begin >= end);
        WARN_ON(!usable_pba(super_ctx, end - 1));

        b = pba_band(super_ctx, begin);

        WARN_ON(b != pba_band(super_ctx, end - 1));

        cb = cache_band(super_ctx, b);

        WARN_ON(free_pbas_in_cache_band(super_ctx, cb) < (end - begin));

        for (i = begin; i < end; ++i)
                super_ctx->pba_map[i] = cb->current_pba++;

        set_bit(band_to_bit(super_ctx, cb, b), cb->map);

        return super_ctx->pba_map[begin];
}

static inline pba_t lookup_pba(struct super_info *super_ctx, pba_t pba)
{
        WARN_ON(!usable_pba(super_ctx, pba));

        return super_ctx->pba_map[pba] == -1 ? pba : super_ctx->pba_map[pba];
}

static inline lba_t lookup_lba(struct super_info *super_ctx, lba_t lba)
{
        return pba_to_lba(lookup_pba(super_ctx, lba_to_pba(lba))) + lba % LBAS_IN_PBA;
}

static void do_free_bio_pages(struct super_info *super_ctx, struct bio *bio)
{
        int32_t i;
        struct bio_vec *bv;

        bio_for_each_segment_all(bv, bio, i) {
                WARN_ON(!bv->bv_page);
                mempool_free(bv->bv_page, super_ctx->page_pool);
                bv->bv_page = NULL;
        }

        /* For now we should only have a single page per bio. */
        WARN_ON(i != 1);
}

static void endio(struct bio *bio)
{
        struct io *io = bio->bi_private;
        struct super_info *super_ctx = io->super_ctx;
        bool rmw_bio = io->bio == NULL;

        if (rmw_bio && bio_data_dir(bio) == WRITE)
                do_free_bio_pages(super_ctx, bio);

        bio_put(bio);

        if (atomic_dec_and_test(&io->pending)) {
                release_io(io, 0);
                complete(&super_ctx->io_completion);
        }
}

static bool adjacent_pbas(struct super_info *super_ctx, pba_t x, pba_t y)
{
        return lookup_pba(super_ctx, x) + 1 == lookup_pba(super_ctx, y);
}

static struct bio *clone_remap_bio(struct io *io, struct bio *bio, int32_t idx,
                                   pba_t pba, int32_t nr_pbas)
{
        struct super_info *super_ctx = io->super_ctx;
        struct bio *clone;

        clone = bio_clone_bioset(bio, GFP_NOIO, super_ctx->bs);
        if (unlikely(!clone)) {
                return NULL;
        }

        if (bio_data_dir(bio) == READ)
                pba = lookup_pba(super_ctx, pba);
        else
                pba = map_pba_range(super_ctx, pba, pba + nr_pbas);

        clone->bi_iter.bi_sector = pba_to_lba(pba);
        clone->bi_private = io;
        clone->bi_end_io = endio;
        clone->bi_bdev = super_ctx->dev->bdev;

        clone->bi_iter.bi_idx = idx;
        clone->bi_vcnt = idx + nr_pbas;
        clone->bi_iter.bi_size = nr_pbas * PBA_SIZE;

        atomic_inc(&io->pending);

        return clone;
}

static void release_bio(struct bio *bio)
{
        struct io *io = bio->bi_private;

        atomic_dec(&io->pending);
        bio_put(bio);
}

static int32_t handle_unaligned_io(struct super_info *super_ctx, struct io *io)
{
        struct bio *bio = io->bio;
        struct bio *clone = bio_clone_bioset(bio, GFP_NOIO, super_ctx->bs);

        WARN_ON(bio_data_dir(bio) != READ);
        WARN_ON(bio_end_lba(bio) > pba_to_lba(bio_begin_pba(bio) + 1));

        if (unlikely(!clone)) {
                return -ENOMEM;
        }

        clone->bi_iter.bi_sector = lookup_lba(super_ctx, bio_begin_lba(bio));
        clone->bi_private = io;
        clone->bi_end_io = endio;
        clone->bi_bdev = super_ctx->dev->bdev;

        atomic_inc(&io->pending);

        super_ctx->tmp_bios[0] = clone;

        return 1;
}

static int32_t split_read_io(struct super_info *super_ctx, struct io *io)
{
        struct bio *bio = io->bio;
        pba_t bp, p;
        int32_t i, n = 0, idx = 0;

        if (unlikely(unaligned_bio(bio)))
                return handle_unaligned_io(super_ctx, io);

        bp = bio_begin_pba(bio);
        p = bp + 1;

        for (i = 1; p < bio_end_pba(bio); ++i, ++p) {
                if (adjacent_pbas(super_ctx, p - 1, p))
                        continue;
                super_ctx->tmp_bios[n] = clone_remap_bio(io, bio, idx, bp, i - idx);
                if (!super_ctx->tmp_bios[n])
                        goto bad;
                ++n, idx = i, bp = p;
        }

        super_ctx->tmp_bios[n] = clone_remap_bio(io, bio, idx, bp, i - idx);
        if (super_ctx->tmp_bios[n])
                return n + 1;

bad:
        while (n--)
                release_bio(super_ctx->tmp_bios[n]);
        return -ENOMEM;
}

static int32_t split_write_io(struct super_info *super_ctx, struct io *io)
{
        struct bio *bio = io->bio;
        int32_t nr_pbas_bio1, nr_pbas_bio2;
        int32_t idx = 0;
        pba_t p;

        nr_pbas_bio1 = pbas_in_band(super_ctx, bio, bio_band(super_ctx, bio));
        nr_pbas_bio2 = pbas_in_bio(bio) - nr_pbas_bio1;
        p = bio_begin_pba(bio);

        super_ctx->tmp_bios[0] = clone_remap_bio(io, bio, idx, p, nr_pbas_bio1);
        if (!super_ctx->tmp_bios[0])
                return -ENOMEM;

        if (!nr_pbas_bio2)
                return 1;

        p += nr_pbas_bio1;
        idx += nr_pbas_bio1;

        super_ctx->tmp_bios[1] =
                clone_remap_bio(io, bio, idx, p, nr_pbas_bio2);
        if (!super_ctx->tmp_bios[1]) {
                release_bio(super_ctx->tmp_bios[0]);
                return -ENOMEM;
        }

        return 2;
}

static int32_t do_sync_io(struct super_info *super_ctx, struct bio **bios, int32_t n)
{
        int32_t i;
        
        if(bios[0]==NULL)
                return -1;

        reinit_completion(&super_ctx->io_completion);

        for (i = 0; i < n; ++i)
                generic_make_request(bios[i]);

        wait_for_completion(&super_ctx->io_completion);

        return atomic_read(&super_ctx->error);
}

typedef int32_t (*split_t)(struct super_info *super_ctx, struct io *io);

static void do_io(struct super_info *super_ctx, struct io *io, split_t split)
{
        int32_t n = split(super_ctx, io);

        if (n < 0) {
                release_io(io, n);
                return;
        }

        WARN_ON(!n);

        do_sync_io(super_ctx, super_ctx->tmp_bios, n);
}

static struct cache_band *cache_band_to_gc(struct super_info *super_ctx, struct bio *bio)
{
        int32_t b = bio_band(super_ctx, bio);
        int32_t nr_pbas = pbas_in_band(super_ctx, bio, b);
        struct cache_band *cb = cache_band(super_ctx, b);

        if (free_pbas_in_cache_band(super_ctx, cb) < nr_pbas)
                return cb;

        if (!usable_band(super_ctx, ++b))
                return NULL;

        cb = cache_band(super_ctx, b);
        nr_pbas = pbas_in_bio(bio) - nr_pbas;

        return free_pbas_in_cache_band(super_ctx, cb) < nr_pbas ? cb : NULL;
}

static struct bio *alloc_bio_with_page(struct super_info *super_ctx, pba_t pba)
{
        struct page *page = mempool_alloc(super_ctx->page_pool, GFP_NOIO);
        struct bio *bio = bio_alloc_bioset(GFP_NOIO, 1, super_ctx->bs);

        if (!bio || !page)
                goto bad;

        bio->bi_iter.bi_sector = pba_to_lba(pba);
        bio->bi_bdev = super_ctx->dev->bdev;

        if (!bio_add_page(bio, page, PAGE_SIZE, 0))
                goto bad;

        return bio;

bad:
        if (page)
                mempool_free(page, super_ctx->page_pool);
        if (bio)
                bio_put(bio);
        return NULL;
}

static void free_rmw_bios(struct super_info *super_ctx, int32_t n)
{
        int32_t i;

        for (i = 0; i < n; ++i) {
                do_free_bio_pages(super_ctx, super_ctx->rmw_bios[i]);
                bio_put(super_ctx->rmw_bios[i]);
        }
}

static bool alloc_rmw_bios(struct super_info *super_ctx, int32_t band)
{
        pba_t p = band_begin_pba(super_ctx, band);
        int32_t i;

        for (i = 0; i < super_ctx->band_size_pbas; ++i) {
                super_ctx->rmw_bios[i] = alloc_bio_with_page(super_ctx, p + i);
                if (!super_ctx->rmw_bios[i])
                        goto bad;
        }
        return true;

bad:
        free_rmw_bios(super_ctx, i);
        return false;
}

static struct bio *clone_bio(struct io *io, struct bio *bio, pba_t pba)
{
        struct super_info *super_ctx = io->super_ctx;
        struct bio *clone = bio_clone_bioset(bio, GFP_NOIO, super_ctx->bs);

        if (unlikely(!clone)) {
                return NULL;
        }

        clone->bi_private = io;
        clone->bi_end_io = endio;
        clone->bi_iter.bi_sector = pba_to_lba(pba);

        atomic_inc(&io->pending);

        return clone;
}

static int32_t do_read_band(struct super_info *super_ctx, int32_t band)
{
        struct io *io = alloc_io(super_ctx, NULL);
        pba_t p = band_begin_pba(super_ctx, band);
        int32_t i;

        if (unlikely(!io))
                return -ENOMEM;

        for (i = 0; i < super_ctx->band_size_pbas; ++i) {
                super_ctx->tmp_bios[i] = clone_bio(io, super_ctx->rmw_bios[i], p + i);
                if (!super_ctx->tmp_bios[i])
                        goto bad;
        }

        return do_sync_io(super_ctx, super_ctx->tmp_bios, super_ctx->band_size_pbas);

bad:
        while (i--)
                release_bio(super_ctx->tmp_bios[i]);
        return -ENOMEM;
}

static int32_t do_modify_band(struct super_info *super_ctx, int32_t band)
{
        struct io *io = alloc_io(super_ctx, NULL);
        pba_t p = band_begin_pba(super_ctx, band);
        int32_t i, j;

        if (unlikely(!io))
            return -ENOMEM;

        for (i = j = 0; i < super_ctx->band_size_pbas; ++i) {
            pba_t pp = lookup_pba(super_ctx, bio_begin_pba(super_ctx->rmw_bios[i]));

            if (pp == p + i)
                continue;

            super_ctx->tmp_bios[j] = clone_bio(io, super_ctx->rmw_bios[i], pp);
            if (!super_ctx->tmp_bios[j])
                goto bad;
            ++j;
        }

        if(!j)
            return -1;

        return do_sync_io(super_ctx, super_ctx->tmp_bios, j);

bad:
        while (j--)
            release_bio(super_ctx->tmp_bios[j]);

        return -ENOMEM;
}

static int32_t do_write_band(struct super_info *super_ctx, int32_t band)
{
        struct io *io = alloc_io(super_ctx, NULL);
        int32_t i;

        if (unlikely(!io))
                return -ENOMEM;

        for (i = 0; i < super_ctx->band_size_pbas; ++i) {
                super_ctx->rmw_bios[i]->bi_private = io;
                super_ctx->rmw_bios[i]->bi_end_io = endio;
                super_ctx->rmw_bios[i]->bi_opf = WRITE;
        }

        atomic_set(&io->pending, super_ctx->band_size_pbas);

        return do_sync_io(super_ctx, super_ctx->rmw_bios, super_ctx->band_size_pbas);
}

static int32_t do_rmw_band(struct super_info *super_ctx, int32_t band)
{
        int32_t r = 0;

        if (!alloc_rmw_bios(super_ctx, band))
                return -ENOMEM;

        r = do_read_band(super_ctx, band);
        if (r < 0)
                goto bad;

        r = do_modify_band(super_ctx, band);
        if (r < 0)
                goto bad;

        return do_write_band(super_ctx, band);

bad:
        printk("do_rmw_band bad!");
        free_rmw_bios(super_ctx, super_ctx->band_size_pbas);
        return r;
}

static void print_cb_info(struct super_info *super_ctx){
        int32_t i;
        printk("Cache Band info:");
        for(i=0;i<super_ctx->nr_cache_bands;i++){
                int32_t fpic = free_pbas_in_cache_band(super_ctx, &super_ctx->cache_bands[super_ctx->cache_bands_map[i]]);
                printk("Cache Band: %d, %d, %p, %d", i, super_ctx->cache_bands_map[i],&super_ctx->cache_bands[super_ctx->cache_bands_map[i]], fpic);
        }
}
static void reset_cache_band(struct super_info *super_ctx, struct cache_band *cb)
{
        //printk("Before reset_cache_band :%p", cb);
        cb->current_pba = cb->begin_pba;
        // if(init_complete)
        //         print_cb_info(super_ctx);
        //printk("Mid reset_cache_band");
        bitmap_zero(cb->map, super_ctx->cache_assoc);
        //printk("After reset_cache_band");
}

// static int32_t do_gc_cache_band(struct super_info *super_ctx, struct cache_band *cb)
// {
//         int32_t i;

//         for_each_set_bit(i, cb->map, super_ctx->cache_assoc) {
//                 int32_t b = bit_to_band(super_ctx, cb, i);
//                 int32_t r = do_rmw_band(super_ctx, b);
//                 if (r < 0)
//                         return r;
//                 unmap_pba_range(super_ctx, band_begin_pba(super_ctx, b), band_end_pba(super_ctx, b));
//         }
//         reset_cache_band(super_ctx, cb);
//         return 0;
// }

static int32_t do_partialGC_every_band(struct super_info *super_ctx, int32_t b)
{
        int32_t r = do_rmw_band(super_ctx, b);
        if (r < 0)
                return r;
        unmap_pba_range(super_ctx, band_begin_pba(super_ctx, b), band_end_pba(super_ctx, b));
        return 0;
}
// get the band list in GC
static void get_partialGC_band_list(struct super_info *super_ctx)
{
        int32_t i,index=0;
        struct cache_band *cb = &super_ctx->cache_bands[super_ctx->cache_bands_map[super_ctx->partialGC_band_index]];
        for_each_set_bit(i, cb->map, super_ctx->cache_assoc) {
                int32_t b = bit_to_band(super_ctx, cb, i);
                super_ctx->partialGC_band_list[index] = b;
                index+=1;
        }
        for(;index<super_ctx->cache_assoc;index++){
                super_ctx->partialGC_band_list[index] = -1;
        }
        super_ctx->current_partialGC_band_list_max = index;
}

static void end_of_partialGC(struct super_info *super_ctx){
        printk("dev: end_of_partialGC");
        super_ctx->partialGC = 0;
        super_ctx->current_partialGC_band_list_index = 0;
        super_ctx->current_partialGC_band_list_max = 0;
        // reset the cache band
        reset_cache_band(super_ctx, &super_ctx->cache_bands[super_ctx->cache_bands_map[super_ctx->partialGC_band_index]]);
}
static void do_partialGC_to_end(struct super_info *super_ctx)
{
        int32_t base = super_ctx->current_partialGC_band_list_index,i;
        for(i=0;i<super_ctx->nr_usable_bands;i++){
                if(super_ctx->partialGC_band_list[base+i]==-1){
                        goto end;
                }
                do_partialGC_every_band(super_ctx, super_ctx->partialGC_band_list[base+i]);
        }
        return;
end:
        end_of_partialGC(super_ctx);
}

static void next_partialGC(struct super_info *super_ctx, struct bio *bio)
{
        int32_t target_cb_index, t;
        int32_t b = bio_band(super_ctx, bio);
        int32_t nr_pbas = pbas_in_band(super_ctx, bio, b);
        struct cache_band *cb = cache_band(super_ctx, b);

        // need GC right now
        if (free_pbas_in_cache_band(super_ctx, cb) < nr_pbas){
                target_cb_index = b % (super_ctx->nr_cache_bands-1);
                printk("dev: next Start_partialGC!");
                reset_cache_band(super_ctx, &super_ctx->cache_bands[super_ctx->cache_bands_map[super_ctx->partialGC_band_index]]);
                t = super_ctx->cache_bands_map[super_ctx->partialGC_band_index];
                super_ctx->cache_bands_map[super_ctx->partialGC_band_index] = super_ctx->cache_bands_map[target_cb_index];
                super_ctx->cache_bands_map[target_cb_index] = t;
                super_ctx->cache_bands[super_ctx->cache_bands_map[target_cb_index]].nr = target_cb_index;
                super_ctx->partialGC = 1;
                get_partialGC_band_list(super_ctx);
                super_ctx->current_partialGC_band_list_index = 0;
                if(init_complete)
                        print_cb_info(super_ctx);
        }
}
static int32_t do_gc_if_required(struct super_info *super_ctx, struct bio *bio)
{
        struct cache_band *cb;
        int32_t r;
        //printk("dev: do_gc_if_required!");
        cb = cache_band_to_gc(super_ctx, bio);
        r = 0;
        if (!cb)
                return r;
        printk("do gc if required 1: %p", cb);
        if(super_ctx->partialGC){
                do_partialGC_to_end(super_ctx);
        }
        next_partialGC(super_ctx,bio);
        return r;  
}

static void start_partialGC(struct super_info *super_ctx)
{
        int32_t t, i=0;
        printk("dev: Start_partialGC!");
        int32_t cache_band_min = super_ctx->band_size_pbas;

        int32_t cahce_band_min_idx = 0;
        for(;i<super_ctx->nr_cache_bands-1;i++){
                int32_t fpic = free_pbas_in_cache_band(super_ctx, &super_ctx->cache_bands[super_ctx->cache_bands_map[i]]);
                if(fpic<cache_band_min){
                        cache_band_min = fpic;
                        cahce_band_min_idx = i;
                }
        }
        reset_cache_band(super_ctx, &super_ctx->cache_bands[super_ctx->cache_bands_map[super_ctx->partialGC_band_index]]);
        t = super_ctx->cache_bands_map[super_ctx->partialGC_band_index];
        super_ctx->cache_bands_map[super_ctx->partialGC_band_index] = super_ctx->cache_bands_map[cahce_band_min_idx];
        super_ctx->cache_bands_map[cahce_band_min_idx] = t;
        super_ctx->cache_bands[super_ctx->cache_bands_map[cahce_band_min_idx]].nr = cahce_band_min_idx;
        super_ctx->partialGC = 1;
        get_partialGC_band_list(super_ctx);
        super_ctx->current_partialGC_band_list_index = 0;
}

static void do_partialGC(struct super_info *super_ctx, int32_t nr_band_to_gc)
{
        int32_t base, i;
        if(nr_band_to_gc==0){
                return;
        }
        if(!super_ctx->partialGC){
                start_partialGC(super_ctx);
        }
        if(init_complete)
                print_cb_info(super_ctx);
        base = super_ctx->current_partialGC_band_list_index;
        for(i=0;i<nr_band_to_gc;i++){
                if(super_ctx->partialGC_band_list[base+i]==-1){
                        goto end;
                }
                do_partialGC_every_band(super_ctx, super_ctx->partialGC_band_list[base+i]);
        }
        super_ctx->current_partialGC_band_list_index += nr_band_to_gc;
        return;
end:
        end_of_partialGC(super_ctx);
}

void get_random_bytes(void *buf, int32_t nbytes);
static int32_t get_random_number(void)
{
    unsigned short randNum;
    get_random_bytes(&randNum, sizeof(unsigned short));
    return randNum;
}

static int32_t get_pc_used_min(struct super_info *super_ctx){
        int32_t i=0, f=0, min = super_ctx->band_size_pbas;
        for(;i<super_ctx->nr_cache_bands-1;i++){
            f = free_pbas_in_cache_band(super_ctx, &super_ctx->cache_bands[super_ctx->cache_bands_map[i]]);
            if(f<min)
                min = f;
                    
        }
        return min;
}

static int32_t get_pc_used_percentage(struct super_info *super_ctx){
        int32_t i=0, f=0;
        for(;i<super_ctx->nr_cache_bands-1;i++){
            f = free_pbas_in_cache_band(super_ctx, &super_ctx->cache_bands[super_ctx->cache_bands_map[i]]);
            if(f<min_pba_in_cache_band)
                return 1;
        }
        return 0;
}

static void dev_d(struct work_struct *work)
{
    struct io* io = container_of(work, struct io, work);
    struct super_info* super_ctx = io->super_ctx;
    struct bio* bio = io->bio;
    struct timeval ct;
    int32_t row, randNum;

    mutex_lock(&super_ctx->lock);

    if (bio_data_dir(bio) == READ) {
        do_io(super_ctx, io, split_read_io);
        prev_io_type = READ;
    }
    else {
        int32_t r;
        WARN_ON(unaligned_bio(bio));
        r = do_gc_if_required(super_ctx, bio);
        if (r < 0){
            release_io(io, r);
        }
        else {
            do_io(super_ctx, io, split_write_io);
            if (RL_GC && get_pc_used_percentage(super_ctx)) {
                do_gettimeofday(&ct);
                io_end_time = ct.tv_sec * 1000 + ct.tv_usec / 1000; //ms
                randNum = get_random_number();
                if (explore_count < 2000) {
                    if ((((randNum + short_max) / (short_max * 2 - 1)) / 100 > epsilon1) || (is_empty_q(&current_state) == 1)) {
                        current_action = get_random_number() % (action_max);
                    }
                    else {
                        current_action = max_action(&current_state);
                    }
                }else {
                    if ((((randNum + short_max) / (short_max * 2 - 1)) / 100 > epsilon2) || (is_empty_q(&current_state) == 1)) {
                        current_action = get_random_number() % (action_max);
                    }
                    else {
                        current_action = max_action(&current_state);
                    }
                }
                explore_count += 1;

                do_partialGC(super_ctx, action_list[current_action]);
                reward = get_reward(io_end_time - io_begin_time);
                next_state = get_next_state(&current_state, current_action, \
                        super_ctx->current_partialGC_band_list_max-super_ctx->current_partialGC_band_list_index,\
                        get_pc_used_min(super_ctx));
                next_state_max_q = max_value_q(&next_state);
                row = get_row_q(&current_state);
                q[row][current_action] = q[row][current_action] + alpha * (reward + gamma * next_state_max_q / 10 - q[row][current_action]) / 10;
                current_state = next_state;
                prev_io_type = WRITE;
            }
        }
    }
    mutex_unlock(&super_ctx->lock);
}

static bool get_args(struct dm_target *ti, struct super_info *super_ctx,
                     int32_t argc, char **argv)
{
        unsigned long long tmp;
        char d;

        if (argc != 5) {
                ti->error = "dm-dev: Invalid argument count.";
                return false;
        }

        if (sscanf(argv[1], "%llu%c", &tmp, &d) != 1 || tmp & 0xfff ||
            (tmp < 4 * 1024 || tmp > 2 * 1024 * 1024)) {
                ti->error = "dm-dev: Invalid track size.";
                return false;
        }
        super_ctx->track_size = tmp;

        if (sscanf(argv[2], "%llu%c", &tmp, &d) != 1 || tmp < 1 || tmp > 200) {
                ti->error = "dm-dev: Invalid band size.";
                return false;
        }
        super_ctx->band_size_tracks = tmp;

        if (sscanf(argv[3], "%llu%c", &tmp, &d) != 1 || tmp < 1 || tmp > 50) {
                ti->error = "dm-dev: Invalid cache percent.";
                return false;
        }
        super_ctx->cache_percent = tmp;

        if (sscanf(argv[4], "%llu%c", &tmp, &d) != 1 ||
            tmp < MIN_DISK_SIZE || tmp > MAX_DISK_SIZE) {
                ti->error = "dm-dev: Invalid disk size.";
                return false;
        }
        super_ctx->disk_size = tmp;

        return true;
}

static void calc_params(struct super_info *super_ctx)
{
        super_ctx->band_size      = super_ctx->band_size_tracks * super_ctx->track_size;
        super_ctx->band_size_pbas = super_ctx->band_size / PBA_SIZE;
        super_ctx->nr_bands       = super_ctx->disk_size / super_ctx->band_size;
        super_ctx->nr_cache_bands = super_ctx->nr_bands * super_ctx->cache_percent / 100 + 1;
        super_ctx->cache_size     = super_ctx->nr_cache_bands * super_ctx->band_size;

        /*
         * Make |nr_usable_bands| a multiple of |nr_cache_bands| so that all
         * cache bands are equally loaded.
         */
        super_ctx->nr_usable_bands  = (super_ctx->nr_bands / (super_ctx->nr_cache_bands-1) - 1) *
                (super_ctx->nr_cache_bands-1);
        super_ctx->cache_assoc    = super_ctx->nr_usable_bands / (super_ctx->nr_cache_bands-1);
        super_ctx->usable_size    = super_ctx->nr_usable_bands * super_ctx->band_size;
        super_ctx->wasted_size    = super_ctx->disk_size - super_ctx->cache_size - super_ctx->usable_size;
        super_ctx->nr_valid_pbas  = (super_ctx->usable_size + super_ctx->cache_size) / PBA_SIZE;
        super_ctx->nr_usable_pbas = super_ctx->usable_size / PBA_SIZE;
        super_ctx->partialGC_band_index = super_ctx->nr_cache_bands-1;
        super_ctx->partialGC = 0;

        WARN_ON(super_ctx->usable_size % PBA_SIZE);
}

static void dev_dtr(struct dm_target *ti)
{
    // Destruct Device
    int32_t i;
    struct super_info *super_ctx = (struct super_info *) ti->private;

    ti->private = NULL;

    if (!super_ctx)
        return;

    vfree(super_ctx->tmp_bios);
    vfree(super_ctx->rmw_bios);
    vfree(super_ctx->pba_map);
    vfree(super_ctx->partialGC_band_list);
    vfree(super_ctx->cache_bands_map);

    for (i = 0; i < super_ctx->nr_cache_bands; ++i)
        if (super_ctx->cache_bands[i].map)
            kfree(super_ctx->cache_bands[i].map);

    vfree(super_ctx->cache_bands);

    mempool_destroy(super_ctx->io_pool);
    destroy_workqueue(super_ctx->queue);
    dm_put_device(ti, super_ctx->dev);
    kzfree(super_ctx);
}

static bool alloc_structs(struct super_info *super_ctx)
{
        int32_t i, size, pba;

        size = sizeof(int32_t) * super_ctx->nr_usable_pbas;
        super_ctx->pba_map = vmalloc(size);
        if (!super_ctx->pba_map)
                return false;
        memset(super_ctx->pba_map, -1, size);

        size = sizeof(int32_t) * super_ctx->nr_usable_bands;
        super_ctx->partialGC_band_list = vmalloc(size);
        if (!super_ctx->partialGC_band_list)
                return false;
        memset(super_ctx->partialGC_band_list, -1, size);

        size = sizeof(struct bio *) * super_ctx->band_size_pbas;
        super_ctx->rmw_bios = vzalloc(size);
        if (!super_ctx->rmw_bios)
                return false;

        super_ctx->tmp_bios = vzalloc(size);
        if (!super_ctx->tmp_bios)
                return false;

        size = sizeof(int32_t) * super_ctx->nr_cache_bands;
        super_ctx->cache_bands_map = vmalloc(size);
        if (!super_ctx->cache_bands_map)
                return false;
        for(i=0;i < super_ctx->nr_cache_bands;i++){
                super_ctx->cache_bands_map[i] = i;
        }

        size = sizeof(struct cache_band) * super_ctx->nr_cache_bands;
        super_ctx->cache_bands = vmalloc(size);
        if (!super_ctx->cache_bands)
                return false;

        /* The cache region starts where the data region ends. */
        pba = super_ctx->nr_usable_pbas;

        size = BITS_TO_LONGS(super_ctx->cache_assoc) * sizeof(long);
        for (i = 0; i < super_ctx->nr_cache_bands; ++i, pba += super_ctx->band_size_pbas) {
                super_ctx->cache_bands[i].nr = i;
                super_ctx->cache_bands[i].begin_pba = pba;
                super_ctx->cache_bands[i].map = kmalloc(size, GFP_KERNEL);
                if (!super_ctx->cache_bands[i].map)
                        return false;
                reset_cache_band(super_ctx, &super_ctx->cache_bands[i]);
        }
        init_complete = 1;
        return true;
}

static int32_t dev_ctr(struct dm_target *ti, unsigned int argc, char **argv)
{
        struct super_info *super_ctx;
        int32_t ret;
        struct timeval ct;

        super_ctx = kzalloc(sizeof(*super_ctx), GFP_KERNEL);
        if (!super_ctx) {
                ti->error = "dm-dev: Cannot allocate dev context.";
                return -ENOMEM;
        }
        ti->private = super_ctx;

        if (!get_args(ti, super_ctx, argc, argv)) {
                kzfree(super_ctx);
                return -EINVAL;
        }

        calc_params(super_ctx);
        // print_params(super_ctx);

        ret = -ENOMEM;
        if (!alloc_structs(super_ctx)) {
            ti->error = "Cannot allocate data structures.";
            goto bad;
        }

        super_ctx->io_pool = mempool_create_slab_pool(16, _io_pool);
        super_ctx->page_pool = mempool_create_page_pool(32, 0);
        super_ctx->bs = bioset_create(16, 0);
        super_ctx->queue = alloc_workqueue("dev_d", WQ_MEM_RECLAIM, 1);

        if (dm_get_device(ti, argv[0], dm_table_get_mode(ti->table), &super_ctx->dev)) {
                ti->error = "dm-dev: Device lookup failed.";
                return -1;
        }

        mutex_init(&super_ctx->lock);
        init_completion(&super_ctx->io_completion);

        ti->num_flush_bios = 1;
        ti->num_discard_bios = 1;
        ti->num_write_same_bios = 1;
        do_gettimeofday(&ct);
        last_io_begin_time = ct.tv_sec*1000+ct.tv_usec/1000;
        return 0;
bad:
	return -EINVAL;
}

static int32_t dev_map(struct dm_target *ti, struct bio *bio)
{
        struct super_info *super_ctx = ti->private;
        struct io *io;
        
        int32_t op = bio_op(bio);
        if (op == REQ_OP_DISCARD || op == REQ_OP_FLUSH) {
                bio->bi_bdev = super_ctx->dev->bdev;
                return DM_MAPIO_REMAPPED;
        }

        io = alloc_io(super_ctx, bio);
        if (unlikely(!io))
                return -EIO;

        queue_io(io);

        return DM_MAPIO_SUBMITTED;
}

static void dev_status(struct dm_target *ti, status_type_t type,
                        unsigned status_flags, char *result, unsigned maxlen)
{
        switch (type) {
        case STATUSTYPE_INFO:
                result[0] = '\0';
                break;

        case STATUSTYPE_TABLE:
        default:
                break;
        }
}

static int32_t dev_iterate_devices(struct dm_target *ti,
                                iterate_devices_callout_fn fn, void *data)
{
        struct super_info *super_ctx = ti->private;

        return fn(ti, super_ctx->dev, 0, ti->len, data);
}


static struct target_type dev_target = {
        .name            = "target_device",      
        .version         = {1, 0, 0},
        .module          = THIS_MODULE,
        .ctr             = dev_ctr,    
        .dtr             = dev_dtr,
        .map             = dev_map,
        .status          = dev_status,
        .prepare_ioctl   = 0,
        .iterate_devices = dev_iterate_devices, 
};

static int32_t __init dev_init(void)
{
        int32_t r;

        _io_pool = KMEM_CACHE(io, 0);
        if (!_io_pool)
                return -ENOMEM;

        r = dm_register_target(&dev_target); 
        if (r < 0) {
                kmem_cache_destroy(_io_pool);
        }

        return r;
}

static void __exit dev_exit(void)
{
        dm_unregister_target(&dev_target);
}

module_init(dev_init);
module_exit(dev_exit);

MODULE_DESCRIPTION(DM_NAME " set-associative disk cache STL emulator target");
MODULE_LICENSE("GPL");
