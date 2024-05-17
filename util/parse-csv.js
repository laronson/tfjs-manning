export async function parseCsv(data) {
  return new Promise((resolve) => {
    data = data.map((row) => {
      return Object.keys(row)
        .sort()
        .map((key) => parseFloat(row[key]));
    });
    resolve(data);
  });
}
