use axum::{
    body::Body,
    http::{Request, Response},
};
use reqwest::header;
use tower::util::ServiceExt;

pub(crate) trait ApiServiceExt<T>
where
    T: tower::Service<Request<Body>>,
{
    fn no_cache(self) -> tower::util::MapResponse<T, fn(T::Response) -> T::Response>;
}

impl<T, ResponseBody> ApiServiceExt<T> for T
where
    T: tower::Service<Request<Body>, Response = Response<ResponseBody>>,
    ResponseBody: http_body::Body,
{
    fn no_cache(self) -> tower::util::MapResponse<T, fn(T::Response) -> T::Response> {
        ServiceExt::<Request<Body>>::map_response(self, |mut record| {
            record.headers_mut().insert(
                header::CACHE_CONTROL,
                header::HeaderValue::from_static("no-cache, no-store"),
            );
            record
                .headers_mut()
                .insert(header::EXPIRES, header::HeaderValue::from_static("-1"));
            record
        })
    }
}
