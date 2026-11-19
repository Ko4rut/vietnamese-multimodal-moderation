import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

// The Next.js helper has only a type import; transpile the real TS source
// to test the same function used by the app without adding a test framework.
const source = readFileSync(
  new URL("../src/features/prediction/utils/model-pair.ts", import.meta.url),
  "utf8",
);
const output = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { getResultModelPair } = await import(
  "data:text/javascript;base64," + Buffer.from(output).toString("base64")
);

const model = (id, classifier) => ({
  id,
  model_name: id,
  classifier,
  validation_macro_f1: null,
  sklearn_version: null,
});

const catalog = [
  model("tfidf_linear_svc_v1_abc", "linear_svc"),
  model("tfidf_logreg_v1_abc", "logistic_regression"),
  model("tfidf_linear_svc_v2_xyz", "linear_svc"),
  model("tfidf_logreg_v2_xyz", "logistic_regression"),
];

test("pairs the matching V2 classifier and preserves tab order on either selection", () => {
  const svc = getResultModelPair(catalog, "tfidf_linear_svc_v2_xyz");
  const lr = getResultModelPair(catalog, "tfidf_logreg_v2_xyz");
  assert.deepEqual(
    svc.map((item) => item.id),
    ["tfidf_linear_svc_v2_xyz", "tfidf_logreg_v2_xyz"],
  );
  assert.deepEqual(lr.map((item) => item.id), svc.map((item) => item.id));
});

test("prefers the matching V1 experiment over the other generation", () => {
  assert.deepEqual(
    getResultModelPair(catalog, "tfidf_logreg_v1_abc").map((item) => item.id),
    ["tfidf_linear_svc_v1_abc", "tfidf_logreg_v1_abc"],
  );
});

test("supports one model or an unknown active ID safely", () => {
  assert.equal(getResultModelPair(catalog.slice(0, 1), catalog[0].id).length, 1);
  assert.deepEqual(getResultModelPair(catalog, "missing"), []);
});

test("falls back to another classifier when no matching version exists", () => {
  const singleVersion = [
    model("svc_v2", "linear_svc"),
    model("lr_v1", "logistic_regression"),
  ];
  assert.deepEqual(
    getResultModelPair(singleVersion, "svc_v2").map((item) => item.id),
    ["svc_v2", "lr_v1"],
  );
});
