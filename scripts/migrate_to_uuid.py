from contextlib import contextmanager
import sys
import sqlite3
from uuid_extensions import uuid7

from logging import getLogger, basicConfig as log_basic_config; log = getLogger(__name__)

#############backing up

def make_backup(con, db_path):
    bak = sqlite3.connect(f"{db_path}.bak")
    con.backup(bak)
    bak.close()

def restore_backup(con, db_path):
    bak = sqlite3.connect(f"{db_path}.bak")
    # con = sqlite3.connect(db_path)
    bak.backup(con)
    bak.close()


############# utilities

def autocommit(f):
    def wrapper(con, *args, **kwargs):
        with con:
            return f(con, *args, **kwargs)
    return wrapper

def list_tables(con):
    return con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

def list_columns(con, table):
    columns = con.execute(f"PRAGMA table_info({table})").fetchall()
    names = [c[1] for c in columns]
    return names

def list_foreign_keys(con, table):
    keys = con.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    return [(key[2:5]) for key in keys]

def get_primary_keys(con, table):
    columns = con.execute(f"PRAGMA table_info({table})").fetchall()
    primary_keys = [c[1] for c in columns if c[5]]
    return primary_keys

############## adding new columns

@autocommit
def add_uuid_column(con: sqlite3.Connection, table: str):
    con.execute(f"ALTER TABLE {table} ADD COLUMN uuid CHAR(32)")
    for rowid, in con.execute(f"SELECT rowid FROM {table}"):
        uuid: str = uuid7().hex # type: ignore
        con.execute(f"UPDATE {table} SET uuid = '{uuid}' WHERE rowid = {rowid}")

@autocommit
def remove_uuid_column(con: sqlite3.Connection, table: str):
    con.execute(f"ALTER TABLE {table} DROP COLUMN uuid")


def add_all_uuid_columns(con):
    for table, in list_tables(con):
        if "id" in list_columns(con, table):
            add_uuid_column(con, table)

def remove_all_uuid_columns(con):
    for table, in list_tables(con):
        if "uuid" in list_columns(con, table):
            remove_uuid_column(con, table)


@autocommit
def foreign_key_id_to_uuid(con, table, column, referenced_table):
    old_end = "_id"
    new_end = "_uuid"
    assert column.endswith(old_end)
    old_column = column
    new_column = column[:-len(old_end)] + new_end

    new_referenced_table = referenced_table
    if referenced_table.endswith("_old"):
        new_referenced_table = referenced_table[:-len("_old")]

    con.execute(f"ALTER TABLE {table} ADD COLUMN {new_column} CHAR(32) REFERENCES {new_referenced_table}(uuid)")
    con.execute(f"UPDATE {table} SET {new_column} = (select uuid from {new_referenced_table} WHERE {new_referenced_table}.id = {old_column})");


def add_all_foreign_keys(con, skip_old_tables=True):
    for table, in list_tables(con):
        if table.endswith("_old") and skip_old_tables: continue
        for referenced_table, column, target_column in list_foreign_keys(con, table):
            if target_column != "id": continue
            foreign_key_id_to_uuid(con, table, column, referenced_table)

def get_new_schema(con, table):
    primary_keys = get_primary_keys(con, table)
    columns = con.execute(f"PRAGMA table_info({table})").fetchall()
    foreign_keys = list_foreign_keys(con, table)

    if len(primary_keys) == 2: # assume it is a many-to-many association table
        assert all(pk.endswith("id") for pk in primary_keys)
        new_primary_keys = [pk[:-len("id")] + "uuid" for pk in primary_keys]
        new_pk = ", ".join(new_primary_keys)
        new_pk = [f"PRIMARY KEY ({new_pk})"]
    elif len(primary_keys) == 1:
        pk= primary_keys[0]
        if pk == "id": new_pk = "uuid"
        else: new_pk = pk
        new_pk = [f"PRIMARY KEY ({new_pk})"]
    elif len(primary_keys) == 0:
        new_pk = []
    else: raise ValueError(f"Table {table} has more than 2 primary keys")

    new_columns = [*columns]
    new_foreign_keys = []

    def drop_column(column_list, column):
        column_index = [c[1] for c in new_columns].index(column)
        column_list.pop(column_index)
    
    try: drop_column(new_columns, "id")
    except ValueError: pass

    for referenced_table, column, target_column in foreign_keys:
        if target_column == "id":
            drop_column(new_columns, column)
        else:
            if referenced_table.endswith("_old"):
                referenced_table = referenced_table[:-len("_old")]
            new_foreign_keys.append(f"FOREIGN KEY ({column}) REFERENCES {referenced_table}({target_column})")

    new_cols = [f"{c[1]} {c[2]}" for c in new_columns]
    new_schema = new_cols + new_foreign_keys + new_pk
    return ", ".join(new_schema)


@contextmanager
def no_foreign_key_check(con):
    with con: con.execute("PRAGMA foreign_keys=off")
    try: yield
    finally: 
        with con: con.execute("PRAGMA foreign_keys=on")

def update_all_table_constraints(con, ):
    tables = list_tables(con)
    with con:
        for table, in tables:
            con.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
    with con:
        for table, in tables:
            old_table = f"{table}_old"
            new_schema = get_new_schema(con, old_table)
            con.execute(f"CREATE TABLE {table} ({new_schema})")
    with no_foreign_key_check(con):
        with con:
            for table, in tables:
                table_columns = ", ".join(list_columns(con, table))
                con.execute(f"INSERT INTO {table} SELECT {table_columns} FROM {table}_old")
        with con:
            for table, in list_tables(con):
                if table.endswith("_old"): continue
                con.execute(f"DROP TABLE {table}_old")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python migrate_to_uuid.py <db_path>")
        sys.exit(1)
    db_path = sys.argv[1]
    log_basic_config(level="DEBUG")
    # db_path = "exp_db_backup.db"
    log.info(f"Using database {db_path}")
    con = sqlite3.connect(db_path)
    con.set_trace_callback(log.info)

    try:
        add_all_uuid_columns(con)
        add_all_foreign_keys(con)
        update_all_table_constraints(con, )
    except:
        log.error("An error occured, restoring backup")
        restore_backup(con, db_path)
        raise
