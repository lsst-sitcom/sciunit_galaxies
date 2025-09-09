import argparse
import logging
import os.path

import astropy.units as u
from lsst.daf.butler.formatters.parquet import astropy_to_arrow, compute_row_group_size, pq
import lsst.daf.butler as dafButler
import lsst.skymap as skymap
from lsst.sitcom.sciunit.galaxies.euclid import _log, query_tract_catalog
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser("make_tract_catalogs")
    parser.add_argument("--log_level", help="Logging level", type=str, default="INFO")
    parser.add_argument("--repo", help="Butler repo to load skymap from", type=str, default="/repo/main")
    parser.add_argument("--save_directory", help="Directory in which tract catalogs will be saved.", type=str)
    parser.add_argument("--skymap_collections", help="Collections to query for skymap", type=str, default="skymaps")
    parser.add_argument("--skymap_name", help="Name of the skymap", type=str, default="lsst_cells_v1")
    parser.add_argument("--tracts", help="Comma-separated list of tracts", type=str, default="5063,4848,4849")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    butler = dafButler.Butler(args.repo)
    skymap = butler.get(skymap.BaseSkyMap.SKYMAP_DATASET_TYPE_NAME, skymap=args.skymap_name, collections=args.skymap_collections)
    tracts = [int(x) for x in args.tracts.split(",")]

    for tract in tracts:
        table, queries, jobs = query_tract_catalog(
            tractInfo=skymap[tract],
            name_skymap=args.skymap_name,
            tmpFile=f"{args.save_directory}/tmp{{idx_patch}}",
        )
        if table is not None:
            table.meta["queries"] = queries
            for column in table.colnames:
                if table[column].unit == u.uJy:
                    table[column] *= 1000.
                    table[column].unit = u.nJy
            for column in ["right_ascension", "declination"]:
                column_error = f"{column}_est_error"
                table[column_error] = np.full(len(table), 0.005/3600, dtype=np.float32)
                table[column_error].description = f"Placeholder {column_error} error (constant 5 mas)"
                table[column_error].unit = u.deg

            directory = f"{args.save_directory}/{tract}"
            if not os.path.isdir(directory):
                os.mkdir(directory)
            tab_arrow = astropy_to_arrow(table)
            row_group_size = compute_row_group_size(tab_arrow.schema)
            filename = f"{directory}/euclid_q1_{tract}_{args.skymap_name}.parq"

            _log.info(f"Writing table for {tract=} to {filename=} with {row_group_size=}")

            pq.write_table(tab_arrow, filename, row_group_size=row_group_size)